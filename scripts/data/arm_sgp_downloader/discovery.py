"""Build the file catalogue from live public listings and test anonymous access."""
import datetime as dt
import json
import re

import requests

from . import config

HEADERS = {"User-Agent": config.USER_AGENT}


def facility_of(filename):
    match = re.match(r"sgp[A-Za-z0-9]*?((?:C|E)\d+)\.[a-z]\d", filename)
    return match.group(1) if match else None


def datastream_of(filename):
    match = re.match(r"(sgp[A-Za-z0-9]+\.[a-z]\d)\.", filename)
    return match.group(1) if match else None


def github_entries(tree):
    entries = []
    for item in tree:
        name = item["path"].rsplit("/", 1)[-1]
        if item["type"] != "blob" or not item["path"].startswith("data/") or not name.endswith((".nc", ".cdf")):
            continue
        stream = datastream_of(name)
        if stream and re.search(config.TARGET_PATTERN, stream) and facility_of(name) in config.CENTRAL_FACILITY_CODES:
            entries.append({
                "source": "arm-test-data", "url": config.GITHUB_RAW_URL + item["path"], "filename": name,
                "datastream": stream, "size": item["size"], "checksum": f"gitblob:{item['sha']}",
            })
    return entries


def static_entries():
    entries = [{"source": "lcrc-arm-diags", "url": config.LCRC_URL + name, "filename": name,
                "datastream": name.removesuffix(".nc"), "size": None, "checksum": None}
               for name in config.LCRC_FILES]
    entries += [{"source": "plumber2", "url": config.PLUMBER2_URL + path, "filename": path.rsplit("/", 1)[-1],
                 "datastream": "FLUXNET2015:US-ARM", "size": None, "checksum": None}
                for path in config.PLUMBER2_FILES]
    return entries


def zenodo_entries(record):
    return [{"source": "zenodo-14261417", "url": f["links"]["self"], "filename": f["key"],
             "datastream": "Ref-EC:E14", "size": f["size"], "checksum": f["checksum"]}
            for f in record["files"] if f["key"] in config.ZENODO_KEYS]


def armlive_credential():
    path = config.ARMLIVE_CREDENTIAL_FILE
    return path.read_text().strip() if path.exists() else None


def armlive_files(stream, credential, last_year):
    """Query year by year because one query returns at most 10000 names."""
    names = []
    for year in range(config.ARMLIVE_FIRST_YEAR, last_year + 1):
        url = (f"{config.ARMLIVE_URL}query?user={credential}&ds={stream}"
               f"&start={year}-01-01&end={year}-12-31&wt=json")
        try:
            reply = requests.get(url, headers=HEADERS, timeout=config.TIMEOUT_S)
        except requests.RequestException as error:
            # The original message holds the URL, and the URL holds the credential.
            raise ConnectionError(f"ARM Live query failed for {stream} {year}: {type(error).__name__}") from None
        if not reply.text.lstrip().startswith("{"):
            raise PermissionError("ARM Live did not accept the credential")
        names += reply.json()["files"]
    return sorted(set(names))


def armlive_entries(credential, last_year):
    return [{"source": f"armlive/{stream}", "url": f"{config.ARMLIVE_URL}saveData?file={name}", "filename": name,
             "datastream": stream, "size": None, "checksum": None, "auth": "armlive"}
            for stream in config.ARMLIVE_STREAMS for name in armlive_files(stream, credential, last_year)]


def arm_inventory():
    query = {"size": 0,
             "query": {"query_string": {"query": "site_code:sgp AND facility_code:(C1 OR E13 OR E14)"}},
             "aggs": {"d": {"terms": {"field": "datastream.keyword", "size": 5000},
                            "aggs": {"s": {"min": {"field": "start_date"}}, "e": {"max": {"field": "end_date"}}}}}}
    reply = requests.post(config.ARM_METADATA_URL, json=query, headers=HEADERS, timeout=config.TIMEOUT_S)
    reply.raise_for_status()
    return [{"datastream": b["key"], "start": b["s"]["value_as_string"][:10], "end": b["e"]["value_as_string"][:10]}
            for b in reply.json()["aggregations"]["d"]["buckets"]
            if re.search(config.TARGET_PATTERN, b["key"]) and "value_as_string" in b["s"]]


def probe_access():
    results = []
    for name, method, url in config.ACCESS_PROBES:
        try:
            reply = requests.request(method, url, headers=HEADERS, timeout=config.TIMEOUT_S, json={} if method == "POST" else None)
            results.append({"route": name, "url": url, "status": reply.status_code,
                            "content_type": reply.headers.get("content-type"), "bytes": len(reply.content)})
        except requests.RequestException as error:
            results.append({"route": name, "url": url, "status": None, "error": type(error).__name__})
    return results


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    tree = requests.get(config.GITHUB_TREE_URL, headers=HEADERS, timeout=config.TIMEOUT_S)
    tree.raise_for_status()
    record = requests.get(config.ZENODO_ECOR_INTERCOMP_URL, headers=HEADERS, timeout=config.TIMEOUT_S)
    record.raise_for_status()
    now = dt.datetime.now(dt.UTC)
    credential = armlive_credential()
    result = {
        "checked_utc": now.isoformat(timespec="seconds"),
        "files": github_entries(tree.json()["tree"]) + static_entries() + zenodo_entries(record.json())
        + (armlive_entries(credential, now.year) if credential else []),
        "arm_inventory": arm_inventory(),
        "access_probes": probe_access(),
    }
    (root / "discovery.json").write_text(json.dumps(result, indent=1))
    return result

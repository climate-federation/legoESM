"""Verified endpoints, datastream targets, and paths. Credentials are read at run time only."""
import os
from pathlib import Path

USER_AGENT = "arm-sgp-downloader"
DEFAULT_ROOT = Path(os.environ.get("ARM_SGP_ROOT", "arm_sgp_data"))
TIMEOUT_S = 120

ARM_METADATA_URL = "https://adc.arm.gov/elastic/metadata/_search"
GITHUB_REPO = "ARM-DOE/arm-test-data"
GITHUB_COMMIT = "fbe0d9ad142d9c8ee7e2a72bd310e0162f977fbd"
GITHUB_TREE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{GITHUB_COMMIT}?recursive=1"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/"
LCRC_URL = "https://web.lcrc.anl.gov/public/e3sm/diagnostics/observations/Atm/arm-diags-data/"
PLUMBER2_URL = "https://thredds.nci.org.au/thredds/fileServer/ks32/CLEX_Data/PLUMBER2/v1-0/"
ZENODO_ECOR_INTERCOMP_URL = "https://zenodo.org/api/records/14261417"

# E13 and E14 are the ARM facility codes co-located at the Lamont Central Facility.
CENTRAL_FACILITY_CODES = ("C1", "E13", "E14")

# Datastream roots in scope, matched against the ARM metadata index (see discovery.py).
TARGET_PATTERN = (
    r"^sgp(1|30|1440)?twr(25m|60m|10x|21x|mr)|^sgp(30)?co2flx(4m|25m|60m)|^sgp30(qc)?ecor|"
    r"^sgpecorsf|^sgp30(ba)?ebbr|^sgp(5|15)ebbr|^sgpsirs|^sgpqcrad|^sgpgndrad|^sgpskyrad|"
    r"^sgpswats|^sgpstamp|^sgpmet|^sgpco2flx(soil|rad|wind|gas)"
)

ARMLIVE_URL = "https://adc.arm.gov/armlive/"
ARMLIVE_CREDENTIAL_FILE = Path.home() / ".arm_live"
ARMLIVE_FIRST_YEAR = 1993
ARMLIVE_WORKERS = 3

# Authenticated route. Raw 10 Hz wind and gas streams (about 470 GB) are deliberately absent.
ARMLIVE_STREAMS = (
    "sgp30twr21xC1.a1", "sgp30twr10xC1.a1", "sgp30twr10xC1.b1", "sgp30twr25mC1.a1", "sgp30twr25mC1.b1",
    "sgp30twr60mC1.a1", "sgp30twr60mC1.b1", "sgp30twrmrC1.c1", "sgp1twrmrC1.c1",
    "sgp30co2flx4mC1.b1", "sgp30co2flx25mC1.b1", "sgp30co2flx60mC1.b1", "sgp30co2flx4mmetC1.b1",
    "sgpco2flx4mC1.b1", "sgpco2flx25mC1.b1", "sgpco2flx60mC1.b1", "sgpco2flxwindC1.b1",
    "sgpco2flxrad4mC1.b1", "sgpco2flxsoilC1.b1", "sgpco2flxsoilauxC1.b1",
    "sgp30ecorE14.b1", "sgp30qcecorE14.c1", "sgpecorsfE14.b1", "sgp30baebbrE13.c1", "sgp30ebbrE13.b1",
    "sgpqcrad1longE13.c2", "sgpqcrad1longE13.c1", "sgpgndrad25m60sC1.b1",
    "sgpswatsE13.b1", "sgpstampE13.b1", "sgpstamppcpE13.b1", "sgpmetE13.b1",
)

LCRC_FILES = ("sgparmdiags1hrC1.c1.nc", "sgparmdiagsdayC1.c1.nc", "sgparmdiagsmonC1.c1.nc")
PLUMBER2_FILES = ("Flux/US-ARM_2003-2012_FLUXNET2015_Flux.nc", "Met/US-ARM_2003-2012_FLUXNET2015_Met.nc")
ZENODO_KEYS = ("Ref-EC-b1-E14.xlsm",)

# Requests that show whether a route is anonymous. No credential is ever attached.
ACCESS_PROBES = (
    ("ARM Live file query", "GET",
     "https://adc.arm.gov/armlive/data/query?ds=sgp30twr60mC1.b1&start=2015-01-01&end=2015-01-02&wt=json"),
    ("ARM metadata index", "GET", "https://adc.arm.gov/elastic/metadata/_search?size=0"),
    ("EOL dataset 71.013 landing page", "GET", "https://data.eol.ucar.edu/dataset/71.013"),
    ("EOL archive path 71.013", "GET", "https://archive.eol.ucar.edu/71.013/"),
    ("AmeriFlux availability list", "GET",
     "https://amfcdn.lbl.gov/api/v1/data_availability/AmeriFlux/BASE-BADM/CCBY4.0"),
    ("AmeriFlux download without identity", "POST", "https://amfcdn.lbl.gov/api/v1/data_download"),
)

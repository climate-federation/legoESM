"""Gap 7/8 controls: NEMO 5.0.1 coefficient laws, not a climate certificate."""
import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import get_policy, set_policy, PrecisionPolicy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, TreguierConfig
from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as gm
from legoesm.ocean.physics.lateral_mixing import _gm_redi_common as common
from scripts.run import run_omip_core2 as runner


@pytest.fixture(autouse=True)
def fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


def _cfg(**kw):
    return GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native",
                       treguier=TreguierConfig(enabled=True, aei0=900),
                       redi_coefficient="nemo21", redi_f_f=jnp.zeros((6, 8)))._replace(**kw)


def _case():
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=400)
    y, x, k = jnp.indices((6, 8, 4))
    T = 16.0 - 2*k + 0.3*jnp.sin(x) + 0.1*y
    S = 35.0 + 0.03*k + 0.01*jnp.cos(x)
    eta = 0.2*jnp.sin(x[..., 0])
    H = jnp.full((6, 8), 400.)
    mask = jnp.ones_like(H)
    um = jnp.ones((6, 9))
    vm = jnp.ones((7, 8)).at[0].set(0).at[-1].set(0)
    ff = 2*constants.Omega*jnp.sin(grid.lat_v[1:])[:, None]*jnp.ones((1,8))
    return (T, S, eta, H, grid, z), dict(mask=mask,u_mask=um,v_mask=vm), _cfg(redi_f_f=ff)


def test_face_law_independent_formula_and_noncommuting_floor():
    # Strong zonal and meridional contrasts straddle the floor; a uniform
    # or all-zero field cannot detect the order of averaging and clipping.
    k = np.array([[0., 100., 800.], [600., 20., 400.], [40., 500., 900.]])
    ft = np.array([[0., .5, -1.], [.25, -.75, 2.], [-.5, 1., -2.]])
    ff = np.roll(ft, 1, axis=0)
    f20 = 2*constants.Omega*np.sin(np.deg2rad(20.))
    u = np.ones_like(k); v = np.ones_like(k)
    u[1, 2] = 0; v[2, 1] = 0
    got = gm.nemo21_redi_from_gm(jnp.array(k),jnp.array(ft*f20),jnp.array(ff*f20),
                                jnp.array(u),jnp.array(v),900.,constants.Omega)
    ku = .5*(k+np.roll(k,-1,axis=1))*u
    kv = .5*(k+np.roll(k,-1,axis=0))*v
    want = ((np.maximum(180.,ku)+(1-np.minimum(1,abs(ft)))*720)*u,
            (np.maximum(180.,kv)+(1-np.minimum(1,abs(ff)))*720)*v)
    for a,b in zip(got,want): np.testing.assert_allclose(a,b,rtol=1e-14)
    assert want[0][0,0] == 900
    assert not np.array_equal(want[0],want[1])
    wrong = .5*(np.maximum(180,k)+np.roll(np.maximum(180,k),-1,axis=1))
    assert wrong[0,1] != max(180,ku[0,1])


def test_face_law_jit_ad_no_artificial_cap_and_sign_symmetry():
    k = jnp.full((2,3),800.)
    f20 = 2*constants.Omega*jnp.sin(jnp.deg2rad(20.))
    f = jnp.full_like(k,.5*f20); m=jnp.ones_like(k)
    fn=lambda a: gm.nemo21_redi_from_gm(a,f,-f,m,m,900.,constants.Omega)[0]
    got=jax.jit(fn)(k)
    np.testing.assert_allclose(got,1160.)  # no post-enhancement cap in NEMO
    np.testing.assert_allclose(jax.grad(lambda a:jnp.sum(fn(a)))(k),1.)
    np.testing.assert_array_equal(*gm.nemo21_redi_from_gm(k,f,-f,m,m,900.,constants.Omega))


@pytest.mark.parametrize('kw,match',[
    ({'redi_coefficient':'typo'},'unknown'),
    ({'slope_scheme':'centered'},'requires'),
    ({'slope_positions':'mode_b'},'requires'),
    ({'treguier':TreguierConfig()},'Treguier'),
    ({'treguier':TreguierConfig(enabled=True,kappa_min=200)},'unfloored'),
    ({'resolution_function':True},'scaling'),
    ({'kappa_redi_lat_scaling':True},'scaling'),
    ({'redi_f_f':None},'exact mesh'),
    ({'redi_aht0':float('nan')},'finite'),
    ({'redi_aht0':0.},'positive'),
    ({'redi_coefficient':'constant'},'redi_f_f'),
])
def test_config_refusals(kw,match):
    with pytest.raises(ValueError,match=match): gm.validate_redi_coefficient(_cfg(**kw))


def test_coriolis_shape_refusal():
    with pytest.raises(ValueError,match='local T grid'):
        gm.nemo21_redi_from_gm(jnp.ones((2,3)),jnp.ones((2,3)),jnp.ones((3,2)),
                             jnp.ones((2,3)),jnp.ones((2,3)),900,constants.Omega)


def test_explicit_and_implicit_consume_same_faces(monkeypatch):
    args, masks, cfg = _case()
    k = jnp.arange(48.).reshape(6,8)*17
    monkeypatch.setattr(gm,'native_treguier_kappa_for_state',lambda *a,**kw:k)
    # Unequal, nonzero signed native slopes expose sign reversal and K33 loss.
    slopes = tuple(jnp.full_like(args[0],v) for v in (-.001,.002,-.003,.004))
    monkeypatch.setattr(gm,'compute_nemo_native_slopes',lambda *a,**kw:slopes)
    expected=gm.nemo21_redi_from_gm(k,jnp.broadcast_to(args[4].f,k.shape),cfg.redi_f_f,
        masks['u_mask'][:,1:],masks['v_mask'][1:],cfg.redi_aht0,constants.Omega)
    seen=[]
    def tensor(q,sx,sy,mask,um,vm,z,jac,grid,ku,act,**kw):
        np.testing.assert_array_equal(ku,expected[0])
        np.testing.assert_array_equal(kw['kappa_Redi_v'],expected[1])
        for a,b in zip(kw['native_slopes'],slopes): np.testing.assert_array_equal(a,b)
        seen.append(q)
        return jnp.zeros_like(q)
    monkeypatch.setattr(gm,'nemo_iso_lap_tracer_tendency_latlon_cgrid',tensor)
    gm.gm_redi_tracer_tendency_latlon(*args,cfg,**masks)
    assert len(seen)==2
    def a33(aht,um,vm,wm,wi,wj,*a,**kw):
        np.testing.assert_array_equal(aht,jnp.broadcast_to(expected[0][...,None],args[0].shape))
        np.testing.assert_array_equal(kw['aht_v'],jnp.broadcast_to(expected[1][...,None],args[0].shape))
        np.testing.assert_array_equal(wi,slopes[2]); np.testing.assert_array_equal(wj,slopes[3])
        return jnp.ones_like(wi),jnp.ones_like(wi)*7
    monkeypatch.setattr(gm,'nemo_iso_a33',a33)
    got=gm.compute_isoneutral_K33_latlon(*args,cfg,**masks)
    np.testing.assert_array_equal(got,jnp.full((6,8,3),7.))


def test_actual_operator_changes_tracers_and_positive_k33():
    args,masks,cfg=_case()
    on=gm.gm_redi_tracer_tendency_latlon(*args,cfg,**masks)
    offcfg=cfg._replace(redi_coefficient='constant',redi_f_f=None)
    off=gm.gm_redi_tracer_tendency_latlon(*args,offcfg,**masks)
    for a,b in zip(on,off):
        assert bool(jnp.isfinite(a).all())
        assert float(jnp.max(jnp.abs(a-b)))>1e-12
    onk=gm.compute_isoneutral_K33_latlon(*args,cfg,**masks)
    offk=gm.compute_isoneutral_K33_latlon(*args,offcfg,**masks)
    assert bool(jnp.isfinite(onk).all()) and bool((onk>=0).all())
    assert float(jnp.max(abs(onk-offk)))>1e-12


@pytest.mark.parametrize('which',['tendency','k33'])
def test_runtime_overrides_raise(which):
    args,masks,cfg=_case()
    fn=gm.gm_redi_tracer_tendency_latlon if which=='tendency' else gm.compute_isoneutral_K33_latlon
    with pytest.raises(ValueError,match='overrides'):
        fn(*args,cfg,kappa_redi_override=jnp.ones((6,8)),**masks)


def test_generic_dry_column_adjoint():
    z=create_ocean_z_star(n_levels=4,H_max=400)
    rho=jnp.broadcast_to(jnp.linspace(1026.,1028.,4),(2,3,4))
    J=jnp.array([[1.,0.,1.],[0.,1.,1.]])
    sx=jnp.full((2,3,3),1e-4); f=jnp.full((2,3),1e-4)
    fn=lambda x:common.compute_treguier_kappa_gm(rho,x,x,z,J,f,TreguierConfig(enabled=True))
    result=jax.jit(fn)(sx)
    assert bool((result[J==0]==0).all())
    grad=jax.grad(lambda x:jnp.sum(fn(x)))(sx)
    assert bool(jnp.isfinite(grad).all())
    assert float(jnp.max(abs(grad)))>0


def test_native_limits_and_nonzero_response():
    args,masks,cfg=_case(); T,S,eta,H,grid,z=args
    rho,J=gm.gm_redi_density_and_jacobian(*args,mask=masks['mask'])
    eos=gm.make_eos_fn('wright')
    w=jnp.zeros_like(T);f=jnp.broadcast_to(grid.f,H.shape).at[2].set(0)
    def fn(x,n2):
        return gm.compute_treguier_kappa_gm_nemo_native(rho,T,S,x,x,masks['mask'],z,grid,f,
            cfg.treguier,eos,jacobian=J,pn2_override=n2,e3w_override=jnp.ones_like(T)*100)
    for n in (0.,-1e-5,1e-5):
        n2=jnp.full_like(T,n)
        out=jax.jit(fn)(w,n2)
        assert bool(jnp.isfinite(out).all())
        assert bool((out[2]==0).all())
        for gradient in jax.grad(lambda x,y:jnp.sum(fn(x,y)),argnums=(0,1))(w,n2):
            assert bool(jnp.isfinite(gradient).all())
    active=fn(w+1e-4,jnp.full_like(T,1e-5))
    assert float(active.max())>0
    np.testing.assert_array_equal(active,fn(w-1e-4,jnp.full_like(T,1e-5)))


CLI=['--grid','tripole','--mesh','unused.nc','--gm-treguier','--gm-kappa-min','0',
     '--gm-slope-scheme','nemo_iso_lap','--gm-slope-positions','nemo_native',
     '--redi-coefficient','nemo21','--redi-aht0','1250']


def test_cli_defaults_and_selection():
    d=runner._build_arg_parser().parse_args([])
    assert d.redi_coefficient is None and d.redi_aht0 is None and d.gm_slope_positions is None
    assert not d.gm_treguier and d.gm_kappa_min==200 and d.gm_aei0==900
    a=runner._build_arg_parser().parse_args(CLI)
    assert a.redi_coefficient=='nemo21' and a.redi_aht0==1250
    assert a.gm_slope_positions=='nemo_native' and a.gm_kappa_min==0
    for flag in ('--redi-coefficient','--gm-slope-positions'):
        with pytest.raises(SystemExit): runner._build_arg_parser().parse_args([flag,'typo'])


@pytest.mark.parametrize('kw',[
    {'redi_coefficient':'typo'}, {'gm_slope_positions':'typo'}, {'grid':'fesom'},
    {'no_gm_redi':True}, {'gm_slope_scheme':'centered'}, {'gm_slope_positions':None},
    {'gm_treguier':False},{'gm_kappa_min':200},{'n_gpus':2},{'distributed':True},
    {'redi_aht0':float('inf')},{'redi_coefficient':None},
])
def test_driver_refusals(kw):
    args=dict(redi_coefficient='nemo21',gm_slope_positions='nemo_native',
        gm_slope_scheme='nemo_iso_lap',gm_treguier=True,gm_kappa_min=0,
        no_gm_redi=False,redi_aht0=900.)
    args.update(kw)
    with pytest.raises(ValueError):runner._validate_omip_redi_selection(**args)


def test_actual_main_validation_and_explicit_default_refusal(monkeypatch):
    for argv,exc in [(['--redi-coefficient','nemo21'],ValueError),
                     (['--gm-aei0','900'],SystemExit),
                     (['--gm-kappa-min=200'],SystemExit)]:
        monkeypatch.setattr('sys.argv',['run',*argv])
        with pytest.raises(exc):runner.main()


def _main_keyword(name,args):
    tree=ast.parse(inspect.getsource(runner.main))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='build_tripole']
    assert len(calls)==1
    expr=next(k.value for k in calls[0].keywords if k.arg==name)
    return eval(compile(ast.Expression(expr),'<actual main>','eval'),dict(args=args))


def test_actual_main_forwarding():
    a=runner._build_arg_parser().parse_args(CLI)
    for name,value in [('redi_coefficient','nemo21'),('redi_aht0',1250),
                       ('gm_slope_positions','nemo_native'),('gm_treguier',True),('gm_kappa_min',0)]:
        assert _main_keyword(name,a)==value


def test_actual_build_configuration(monkeypatch,tmp_path):
    import xarray as xr
    from legoesm.ocean.fidelity.nemo_match_recipe import NEMOMatchTripoleRecipeConfig,nemo_match_tripole_model_config
    base=nemo_match_tripole_model_config(NEMOMatchTripoleRecipeConfig())
    lat=np.arange(48.).reshape(6,8)-24
    mesh=tmp_path/'mesh.nc';xr.Dataset({'gphif':(('y','x'),lat)}).to_netcdf(mesh)
    tree=ast.parse(inspect.getsource(runner.build_tripole))
    # Execute actual new config block, including its real mesh read and assignment.
    blocks=[n for n in tree.body[0].body if isinstance(n,ast.If)
            and ast.unparse(n.test)=='redi_coefficient is not None or gm_slope_positions is not None']
    assert len(blocks)==1
    args,masks,cfg=_case()
    env=dict(vars(runner),redi_coefficient='nemo21',gm_slope_positions='nemo_native',
        redi_aht0=1250.,mesh_path=str(mesh),grid=args[4],config=base,
        _ovr={'gm_redi':runner._tripole_treguier_gm_redi(900.,0.)._replace(slope_scheme='nemo_iso_lap')})
    exec(compile(ast.Module(body=blocks,type_ignores=[]),'<actual build>','exec'),env)
    got=env['_ovr']['gm_redi']
    assert got.redi_coefficient=='nemo21' and got.redi_aht0==1250
    assert got.slope_positions=='nemo_native'
    np.testing.assert_allclose(got.redi_f_f,2*base.omega*np.sin(np.deg2rad(lat)),rtol=1e-14)
    gm.validate_redi_coefficient(got)


def test_build_early_guard():
    with pytest.raises(ValueError,match='unknown'):
        runner.build_tripole(4,400,'unused.nc',redi_coefficient='typo')


def test_k33_model_rotation_forwarding():
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model
    tree=ast.parse(inspect.getsource(model))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='compute_isoneutral_K33_latlon']
    assert len(calls)==2
    for c in calls:
        value=next(k.value for k in c.keywords if k.arg=='omega')
        assert eval(compile(ast.Expression(value),'<production K33>','eval'),
                    {'_cfg_b':SimpleNamespace(omega=1.23e-4)})==1.23e-4

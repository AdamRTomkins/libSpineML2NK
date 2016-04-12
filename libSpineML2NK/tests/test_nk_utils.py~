def test_fixed_value():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    
    prop = smlNetwork.PropertyType('C','S',smlNetwork.FixedValueType(0.123),None)    
    assert nk_utils.gen_value(prop,0) == 0.123

def test_fixed_value_units():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    
    prop = smlNetwork.PropertyType('C','mS',smlNetwork.FixedValueType(1000),None)    
    assert nk_utils.gen_value(prop,0) == 1

def test_fixed_value_list():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    prop = smlNetwork.PropertyType('C','S',smlNetwork.ValueListType([1,2,3]),None)
    assert nk_utils.gen_value(prop,1) == 2


def test_uniform_distrobution_list():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    import numpy as np
    prop = smlNetwork.PropertyType('C','S',None,smlNetwork.UniformDistributionType(1,2,3))
    tmp = nk_utils.gen_value(prop,0)    
    np.random.seed(1) 
    assert tmp == np.random.uniform(2,3)

def test_normal_distrobution_list():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    import numpy as np
    prop = smlNetwork.PropertyType('C','S',None,smlNetwork.NormalDistributionType(1,2,3))    
    tmp = nk_utils.gen_value(prop,0)    
    np.random.seed(1) 
    assert tmp == np.random.normal(2,3)

def test_poisson_distrobution_list():
    from libSpineML import smlNetwork
    from libSpineML2NK import nk_utils
    import numpy as np
    prop = smlNetwork.PropertyType('C','S',None,smlNetwork.PoissonDistributionType(1,2))    
    tmp = nk_utils.gen_value(prop,0)
    np.random.seed(1) 
    assert  tmp == np.random.poisson(2)

####################################
#
####################################

def test_TimeVaryingInput():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment

    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable


    exp = smlExperiment('examples/experiment0.xml')

    inputs = np.zeros((100, 1), dtype=np.double)
    time = np.arange(0, 100, 1)
    
    lpu_size    = 1
    lpu_start   = 0

    params = e.bundle.experiments[0].Experiment[0].AbstractInput[0]

    I = nk_utils.TimeVaryingInput(params,lpu_start,lpu_size,time,inputs):

    assert  tmp == np.random.poisson(2)





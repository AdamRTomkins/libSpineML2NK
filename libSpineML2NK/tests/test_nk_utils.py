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

    tp1 = smlExperiment.TimePointValueType(0,300)

    tp2 = smlExperiment.TimePointValueType(0,300)

    tvit = smlExperiment.TimeVaryingInputType(start_time = 0,duration = 1,TimePointValue = [tp1,tp2])
    
    inputs = np.zeros((100, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.TimeVaryingInput(tvit,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == sum(300 * np.ones((100, 1), dtype=np.double))

def test_TimeVaryingInput_Spike():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment

    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable

    tp1 = smlExperiment.TimePointValueType(0)
    tp2 = smlExperiment.TimePointValueType(0.5)

    tvit = smlExperiment.TimeVaryingInputType(start_time = 0,duration = 1,TimePointValue = [tp1,tp2])
    
    inputs = np.zeros((100, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.TimeVaryingInput(tvit,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == 2 
    assert  sum(I)/100 == 0.02

def test_TimeVaryingInput_RateBased():
    assert False



def test_ConstantArrayInput():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment

    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable


    cai = smlExperiment.ConstantArrayInputType(start_time = 0,duration = 1, array_size = 1, array_value = [1])

    length = 100
    inputs = np.zeros((length, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.ConstantArrayInput(cai,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == length    

def test_ConstantInput():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment

    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable


    ci = smlExperiment.ConstantInputType(start_time = 0,duration = 1, value = 1)

    length = 100
    inputs = np.zeros((length, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.ConstantInput(ci,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == length    


def test_TimeVaryingArrayInput():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment
    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable

    tva1 = smlExperiment.TimePointArrayValueType(array_time = [0,0.5],array_value = [2,0],index=0)

    tvait = smlExperiment.TimeVaryingArrayInputType(start_time = 0,duration = 1,array_size = 2, TimePointArrayValue = [tva1])

    inputs = np.zeros((100, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.TimeVaryingArrayInput(tvait,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == 100

def test_TimeVaryingArrayInput_Spike():
    from libSpineML import smlNetwork
    from libSpineML import smlExperiment
    from libSpineML2NK import nk_utils
    import numpy as np
    from libSpineML2NK import nk_executable

    tva1 = smlExperiment.TimePointArrayValueType(array_time = [0,0.5],index=0)

    tvait = smlExperiment.TimeVaryingArrayInputType(start_time = 0,duration = 1,array_size = 2, TimePointArrayValue = [tva1])

    inputs = np.zeros((100, 1), dtype=np.double)
    time = np.arange(0, 1, 0.01)
    
    lpu_size    = 1
    lpu_start   = 0

    I = nk_utils.TimeVaryingArrayInput(tvait,lpu_start,lpu_size,time,inputs)

    assert  sum(I) == 2


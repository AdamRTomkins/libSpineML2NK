def test_instatiation():
    import libSpineML2NK
    from libSpineML2NK import nk_executable
    e = nk_executable.Executable()
    assert type(e) is nk_executable.Executable


def test_process_experiment():
    import libSpineML2NK
    from libSpineML2NK import nk_executable
    e = nk_executable.Executable('examples/experiment0.xml')
    e.process_experiment(0,0)
    assert e.params['name'] == 'Constant'
    assert e.params['steps'] == 1000
    assert e.params['dt'] == 0.0001



def test_process_network():
    import libSpineML2NK
    from libSpineML2NK import nk_executable
    e = nk_executable.Executable('examples/experiment0.xml')
    e.process_network()
    

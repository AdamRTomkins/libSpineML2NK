class Component:
    """ Common class for SpineML components """
    def __init__(self,ctype):
        self.type = ctype # Currently Unused
        self.initial_regime = 0
        self.regimes = []
        self.parameters = []
        self.state_variables = []
        self.ports = []
        self.num_regimes = 0
        self.send_ports = []
        self.recieve_ports = []
        
    def add_regime(self,regime,initial):
        self.regimes.append({"index":self.num_regimes,"regime":regime})
        if initial == True:
            self.initial_regime = self.num_regimes
        self.num_regimes += 1

    def add_state_variable(self,name,dimension):
        self.state_variables.append({"name":name,"dimension":dimension})

    def add_parameter(self,name,dimension):
        self.parameters.append({"name":name,"dimension":dimension})

    def add_port(self,port,name):           
        self.ports.append({port:name})

    def add_send_port(self,port_type,name): # NK: Store output variable
        self.send_ports.append({"name":name,"type":port_type}) # Neurokernel can only have one!

    def add_recieve_port(self,port_type,name,dimension="?",op="+"): # NK:Store Input Variable
        self.recieve_ports.append({"name":name,"type":port_type,"dimension":dimension,"op":op})

    def set_initial_regime(self,initial):
        self.initial_regime = ""



class Regime:       
    """ Common class for SpineML component regimes """

    def __init__(self,name,condition=None,math=None):
        self.name = name
        if condition is not None:        
            self.conditions = [condition]
        else:
            self.conditions = []
        if math is not None:        
            self.math = [math]
        else:
            self.math = []

    def add_derivative(self,value,time_derivative):
        self.math.append({"parameter":value,"math":time_derivative})
    
    def add_condition(self,condition):
        self.conditions.append(condition)
  

class Condition:
    """ Common class for SpineML component regime condition """
    regime = ""
    assignments = []
    trigger = ""
    events = []

    def __init__(self,regime):
        self.regime = regime
        self.assignments = []
        self.trigger = []
        self.events = []

    def add_assignment(self,variable,math):
        """ add state assignment """
        self.assignments.append({"parameter":variable,"math":math})
        
    def add_trigger(self,math):
        """ add trigger to a condition """
        self.trigger = math
    
    def add_event(self,port):
        """ add event port to a condition """
        self.events.append(port)

    




### Test Class
### Need to integrate XSLT componants 
"""
con = Condition("integrating")
con.add_assignment("V","Vr")
con.add_assignment("output","1")
con.add_trigger("V > Vt")

reg = Regime("integrating")
reg.add_derivative("V","((I_Syn) / C) + (Er - V) / (R*C)")   # Uses i to match code, fix
reg.add_condition(con)                                   # add dimentionality

com = Component()

com.add_regime(reg,True)

com.add_state_variable("V","mV")
com.add_state_variable("output","?")
com.add_send_port("Event","spike")

com.add_parameter("C","nS")
com.add_parameter("Vt","mV")
com.add_parameter("Er","mV")
com.add_parameter("Vr","mV")
com.add_parameter("R","MOhm")
"""
######################################################
# SpineML Specifyable Neuron
######################################################




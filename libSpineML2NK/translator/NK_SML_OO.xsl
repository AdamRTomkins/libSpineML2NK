<?xml version="1.0" encoding="windows-1252"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:EX="http://www.shef.ac.uk/SpineMLExperimentLayer"
xmlns:NL="http://www.shef.ac.uk/SpineMLNetworkLayer"
xmlns:LL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer"
xmlns:CL="http://www.shef.ac.uk/SpineMLComponentLayer"
xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>

<xsl:template match="/">

<xsl:comment>          <!-- SpineML Translator for Neurokernel
                    This is a work in progress which currently only supports a fraction of the SpineML specification

    Supported:
        Networks
            :: Assumes all neurons are LIF neurons
            :: Assumes neurons are specific to the NK formulation ( Cm and R not Tau)
            :: No checking to force other wise
        Inputs
            :: Supports All inputs, but requires testing
        Populations
            :: Rolls populations into one LPU using constructors to allow for different 'populations'
    Future Plans

        SpineML Spec
            :: Componant Generation
            :: Lesions and Congifurations

        Python Translations
            :: Dynamic Neuron Checking
            :: Dynamic file name creation not just SpineML, use the experiment name
 -->
</xsl:comment>

# Imports here
import pdb
import argparse
import itertools

from neurokernel.core_gpu import Module

import numpy as np

import networkx as nx
import h5py

import cPickle
import scipy
from scipy import io

import os
import sys
import json

import translator_files

from translator_files.LPU import LPU

from translator_files.translator import nk_spineml
from translator_files.translator import nk_component
#from nk_spineml import Experiment

import neurokernel.mpi_relaunch
# Neurokernal Experiment Object

<xsl:for-each select="/EX:SpineML/EX:Experiment">
exp = nk_spineml.Experiment('<xsl:value-of select="@name"/>','<xsl:value-of select="@description"/>')

        <xsl:apply-templates select="EX:Simulation"/>

        <xsl:apply-templates select="EX:Model"/>

            <xsl:for-each select="EX:ConstantInput">
                <xsl:message terminate="no">
Warning: ConstantInput Data Detected, Currently In Progress
                </xsl:message>

exp.add_input({'type':'ConstantInput',
            <xsl:choose>
                <xsl:when test="@name">
                'name':'<xsl:value-of select="@name"/>',
                </xsl:when>
                <xsl:otherwise>'name':'input name',</xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target">
                'target':'<xsl:value-of select="@target"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires target.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target_indicies">
                'target_indicies':'<xsl:value-of select="@target_indicies"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> lacks target_indicies, assuming target population.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@port">
                'port':'<xsl:value-of select="@port"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires port.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@start_time">
                'start_time':<xsl:value-of select="@start_time"/>,
                </xsl:when>
                <xsl:otherwise>
                'start_time':0,
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@duration">
                'duration':'<xsl:value-of select="@duration"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@rate_based_distribution">
                'rate_based_distribution':'<xsl:value-of select="@rate_based_distribution"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@value">
                'value':<xsl:value-of select="@value"/>
                </xsl:when>
            </xsl:choose>
		}) ##
            </xsl:for-each>

            <xsl:for-each select="EX:ConstantArrayInput">
                <xsl:message terminate="no">
Warning: ConstantArrayInput Data Detected, Currently in progress.
                </xsl:message>

exp.add_input({'type':'ConstantArrayInput',
                <xsl:choose>
                <xsl:when test="@name">
                'name':'<xsl:value-of select="@name"/>',
                </xsl:when>
                <xsl:otherwise>'name':'input name',</xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target">
                'target':'<xsl:value-of select="@target"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires target.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target_indicies">
                'target_indicies':'<xsl:value-of select="@target_indicies"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> lacks target_indicies, assuming target population.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@port">
                'port':'<xsl:value-of select="@port"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires port.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@start_time">
                'start_time':<xsl:value-of select="@start_time"/>,
                </xsl:when>
                <xsl:otherwise>
                'start_time':0,
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@duration">
                'duration':'<xsl:value-of select="@duration"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@rate_based_distribution">
                'rate_based_distribution':'<xsl:value-of select="@rate_based_distribution"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@array_size">
                'array_size':<xsl:value-of select="@array_size"/>,
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> has no array_size, defaults to 0
                    </xsl:message>
                'array_size':0,
                </xsl:otherwise>
            </xsl:choose>
<xsl:choose>
                <xsl:when test="@array_value">
                'array_value':[<xsl:value-of select="@array_value"/>]
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Warning: <xsl:value-of select="@name"/>  has no array_value
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
                    }) ###

            </xsl:for-each>

            <xsl:for-each select="EX:TimeVaryingInput">
                <xsl:message terminate="no">
Error: <xsl:value-of select="@name"/> Data Detected, Currently in development
                </xsl:message>

exp.add_input({'type':'TimeVaryingInput',
            <xsl:choose>
                <xsl:when test="@name">
                'name':'<xsl:value-of select="@name"/>',
                </xsl:when>
                <xsl:otherwise>'name':'input name',</xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target">
                'target':'<xsl:value-of select="@target"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires target.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target_indicies">
                'target_indicies':'<xsl:value-of select="@target_indicies"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> lacks target_indicies, assuming target population.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@port">
                'port':'<xsl:value-of select="@port"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires port.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@start_time">
                'start_time':<xsl:value-of select="@start_time"/>,
                </xsl:when>
                <xsl:otherwise>
                'start_time':0,
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@duration">
                'duration':'<xsl:value-of select="@duration"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@rate_based_distribution">
                'rate_based_distribution':'<xsl:value-of select="@rate_based_distribution"/>',
                </xsl:when>
            </xsl:choose>
                'TimePointValue':[<xsl:for-each select="EX:TimePointValue">
                    {<xsl:choose>
                    <xsl:when test="@value">
                            'value':<xsl:value-of select="@value"/>,
                    </xsl:when>
                    <xsl:otherwise><xsl:message terminate="no">Error: <xsl:value-of select="@name"/> has no value, spike times assumed</xsl:message></xsl:otherwise></xsl:choose>
                        <xsl:choose>
                    <xsl:when test="@time">
                            'time':<xsl:value-of select="@time"/>
                    </xsl:when>
                    <xsl:otherwise><xsl:message terminate="yes">
        Error:<xsl:value-of select="@name"/> has no time supplied
                    </xsl:message></xsl:otherwise>
                </xsl:choose>
      		     }<xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose>
            </xsl:for-each>
                ]})#End Input
            </xsl:for-each>

            <xsl:for-each select="EX:TimeVaryingArrayInput">
exp.add_input({'type':'TimeVaryingArrayInput',
                <xsl:choose>
                <xsl:when test="@name">
                'name':'<xsl:value-of select="@name"/>',
                </xsl:when>
                <xsl:otherwise>'name':'input name',</xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target">
                'target':'<xsl:value-of select="@target"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires target.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@target_indicies">
                'target_indicies':'<xsl:value-of select="@target_indicies"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> lacks target_indicies, assuming target population.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@port">
                'port':'<xsl:value-of select="@port"/>',
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> requires port.
                    </xsl:message>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@start_time">
                'start_time':<xsl:value-of select="@start_time"/>,
                </xsl:when>
                <xsl:otherwise>
                'start_time':0,
                </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@duration">
                'duration':'<xsl:value-of select="@duration"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@rate_based_distribution">
                'rate_based_distribution':'<xsl:value-of select="@rate_based_distribution"/>',
                </xsl:when>
            </xsl:choose>
            <xsl:choose>
                <xsl:when test="@array_size">
                'array_size':<xsl:value-of select="@array_size"/>,
                </xsl:when>
                <xsl:otherwise>
                    <xsl:message terminate="no">
Warning: <xsl:value-of select="@name"/> has no array_size, defaults to 0
                    </xsl:message>
                'array_size':0,
                </xsl:otherwise>
            </xsl:choose>
    		'TimePointArrayValue':[
                    <xsl:for-each select="EX:TimePointArrayValue">{
                        <xsl:choose>
                    <xsl:when test="@index">
                            'index':<xsl:value-of select="@index"/>,
                    </xsl:when>
                    <xsl:otherwise><xsl:message terminate="yes">
        Error: TimePointArrayValue has no index
                    </xsl:message></xsl:otherwise>
                </xsl:choose>
                <xsl:choose>
                    <xsl:when test="@array_time">
                            'array_time':[<xsl:value-of select="@array_time"/>],
                    </xsl:when>
                    <xsl:otherwise><xsl:message terminate="yes">
                        Error: TimePointArrayValue has no array_time
                    </xsl:message></xsl:otherwise>
                </xsl:choose>
                <xsl:choose>
                    <xsl:when test="@array_value">
                            'array_value':[<xsl:value-of select="@array_value"/>]
                    </xsl:when>
                    <xsl:otherwise><xsl:message terminate="yes">
Error: <xsl:value-of select="@name"/> has no array_value, assuming a spiking componant has been specified, which is not yet supported
                    </xsl:message></xsl:otherwise>
                </xsl:choose>
                            } <xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose>
                            </xsl:for-each>
                        ]
                    }) #
           </xsl:for-each>
           </xsl:for-each>

exp.run()
import matplotlib.pyplot as plt; import h5py
f = h5py.File('spine_ml_V.h5')['array']; plt.subplot(311);plt.plot(f);
f = h5py.File('spine_ml_I.h5')['array']; plt.subplot(312);plt.plot(f);

plt.show()
</xsl:template>


<xsl:template match="EX:Model">
        <xsl:variable name="network_layer" select="document(@network_layer_url)"/>
         <xsl:for-each select="$network_layer/LL:SpineML/LL:Population">


            <xsl:variable name="pop_name" select="translate(LL:Neuron/@name,' ', '_')"/>
            <xsl:variable name="pop_size" select="translate(LL:Neuron/@size,' ', '_')"/>
            <xsl:variable name="pop_url" select="translate(LL:Neuron/@url,' ', '_')"/>

            <xsl:variable name="component" select="document(LL:Neuron/@url)"/>
    
            <xsl:variable name="component_file" select="LL:Neuron/@url"/>

# Define Components
  
<xsl:for-each select="$component/CL:SpineML/CL:ComponentClass">

<xsl:variable name="com_name" select="translate(@name,' ', '_')"/>

<xsl:value-of select="$com_name"/> = nk_component.Component("<xsl:value-of select="@type"/>")
# Dynamics
<xsl:for-each select="CL:Dynamics">

<xsl:variable name="initial_regime" select="@initial_regime"/>

<xsl:for-each select="CL:StateVariable">
<xsl:value-of select="$com_name"/>.add_state_variable("<xsl:value-of select="@name"/>","<xsl:value-of select="@dimension"/>")
</xsl:for-each>

<xsl:for-each select="CL:Regime">
<xsl:variable name="reg_name" select="translate(@name,' ', '_')"/>
<xsl:value-of select="$reg_name"/> = nk_component.Regime("<xsl:value-of select="@name"/>")

# Time Derivatives
<xsl:for-each select="CL:TimeDerivative">
<xsl:value-of select="$reg_name"/>.add_derivative("<xsl:value-of select="@variable"/>","<xsl:value-of select="CL:MathInline"/>")
</xsl:for-each>

# OnConditions are named using random strings to allow multiple conditions per target regime
<xsl:for-each select="CL:OnCondition">
<xsl:variable name="con_name" select="concat(translate(@name,' ', '_'),generate-id())"/>
<xsl:value-of select="$con_name"/> = nk_component.Condition("<xsl:value-of select="@target_regime"/>")

#con.add_assignment("V","Vr")
<xsl:for-each select="CL:StateAssignment">
<xsl:value-of select="$con_name"/>.add_assignment("<xsl:value-of select="@variable"/>","<xsl:value-of select="CL:MathInline"/>")
</xsl:for-each>

#con.add_trigger("V > Vt")
<xsl:for-each select="CL:Trigger">
<xsl:value-of select="$con_name"/>.add_trigger("<xsl:value-of select="CL:MathInline"/>")
</xsl:for-each>


#con.add_event("spike")
<xsl:for-each select="CL:EventOut">
<xsl:value-of select="$con_name"/>.add_event("<xsl:value-of select="@port"/>")
</xsl:for-each>

<xsl:value-of select="$reg_name"/>.add_condition(<xsl:value-of select="$con_name"/>)      

</xsl:for-each>

<xsl:value-of select="$com_name"/>.add_regime(<xsl:value-of select="$reg_name"/>,
<xsl:choose>
  <xsl:when test="$reg_name = $initial_regime">
    True
  </xsl:when>
  <xsl:otherwise>
    False
  </xsl:otherwise>
</xsl:choose>)

</xsl:for-each>
</xsl:for-each>


<xsl:for-each select="CL:Parameter">
<xsl:value-of select="$com_name"/>.add_parameter("<xsl:value-of select="@name"/>","<xsl:value-of select="@dimension"/>")
</xsl:for-each>

<xsl:for-each select="CL:AnalogSendPort">
<xsl:value-of select="$com_name"/>.add_send_port("Analog","<xsl:value-of select="@name"/>")
</xsl:for-each>
<xsl:for-each select="CL:EventSendPort">
<xsl:value-of select="$com_name"/>.add_send_port("Event","<xsl:value-of select="@name"/>")
</xsl:for-each>
<xsl:for-each select="CL:AnalogReducePort">
<xsl:value-of select="$com_name"/>.add_recieve_port("AnalogReduce","<xsl:value-of select="@name"/>","<xsl:value-of select="@dimension"/>","<xsl:value-of select="@reduce_op"/>")
</xsl:for-each>
exp.add_component("<xsl:value-of select="$component_file"/>",<xsl:value-of select="$com_name"/>)
</xsl:for-each>


model_params = {
                    <xsl:for-each select="LL:Neuron/*">
                        <xsl:variable name="n_name" select="translate(@name,' ', '_')"/>
                        <xsl:variable name="n_dimension" select="translate(@dimension,' ', '_')"/>
                        '<xsl:value-of select="$n_name"/>': {
                        'dimension': '<xsl:value-of select="$n_dimension"/>',
                        <xsl:for-each select="*">
                        'input':{'type':'<xsl:value-of select="name()"/>',
                            <xsl:if test="name() = 'FixedValue'">'value':<xsl:value-of select="@value"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'ValueList'">
                                <xsl:message terminate="yes"> Parameter type  <xsl:value-of select="name()"/> used, currently not supported </xsl:message>
                            </xsl:if>
                            <xsl:if test="name() = 'UniformDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'maximum':<xsl:value-of select="@maximum"/>,
                                'minimum':<xsl:value-of select="@minimum"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'NormalDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'mean':<xsl:value-of select="@mean"/>,
                                'variance':<xsl:value-of select="@variance"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'PoissonDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'mean':<xsl:value-of select="@mean"/>}
                            </xsl:if>
                         </xsl:for-each>}<xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose>
                </xsl:for-each>
                }
exp.add_population('<xsl:value-of select="$pop_name"/>',<xsl:value-of select="$pop_size"/>,'<xsl:value-of select="$pop_url"/>', model_params)


                <xsl:for-each select="LL:Projection">
synapse_list = [

                    <xsl:for-each select="LL:Synapse">
                    {
                        <xsl:apply-templates select="NL:AllToAllConnection"/>
                        <xsl:apply-templates select="NL:OneToOneConnection"/>
                        <xsl:apply-templates select="NL:FixedProbabilityConnection"/>
                        <xsl:apply-templates select="NL:ConnectionList"/>
                        <xsl:apply-templates select="LL:WeightUpdate"/>
                        <xsl:apply-templates select="LL:PostSynapse"/>
                    }<xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose>
                    </xsl:for-each>

]
exp.add_projection('<xsl:value-of select="$pop_name"/>', '<xsl:value-of select="@dst_population"/>', synapse_list)
                </xsl:for-each>



            </xsl:for-each>



    <xsl:apply-templates select="Lesion"/>
    <xsl:apply-templates select="Configuration"/>



</xsl:template>



<xsl:template match="NL:OneToOneConnection">
         'type': 'OneToOneConnection',
         'delay_dimension': '<xsl:value-of select="NL:Delay/@Dimension" />',
         'delay': <xsl:value-of select="NL:Delay/NL:FixedValue/@value" />,
</xsl:template>

<xsl:template match="NL:AllToAllConnection">
         'type': 'AllToAllConnection',
         'delay_dimension': '<xsl:value-of select="NL:Delay/@Dimension" />',
         'delay': <xsl:value-of select="NL:Delay/NL:FixedValue/@value" />,
</xsl:template>

<xsl:template match="NL:FixedProbabilityConnection">
         'type': 'FixedProbabilityConnection',
         'delay_dimension': '<xsl:value-of select="NL:Delay/@Dimension" />',
         'delay': <xsl:value-of select="NL:Delay/NL:FixedValue/@value" />,
         'probability': <xsl:value-of select="@probability" />,
         'seed': <xsl:value-of select="@seed" />,
</xsl:template>

<xsl:template match="NL:ConnectionList">
         'type': 'ConnectionList',
         'connections' : [
                           <xsl:for-each select="*">[<xsl:value-of select="@src_neuron"/>,<xsl:value-of select="@dst_neuron" />,<xsl:value-of select="@delay"/>]<xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose></xsl:for-each>
                         ],
</xsl:template>

<xsl:template match="LL:WeightUpdate">
        'weightupdate' : {
            'name': '<xsl:value-of select="@name" />',
            'url': '<xsl:value-of select="@url" />',
            'input_src_port': '<xsl:value-of select="@input_src_port" />',
            'input_dst_port': '<xsl:value-of select="@input_dst_port" />'
                    <xsl:message terminate="no">
Warning: Weight Update not supported, continuing with static weights = 1
                    </xsl:message>
                     },
</xsl:template>

<xsl:template match="LL:PostSynapse">
        'postsynapse' : {
            'name': '<xsl:value-of select="@name" />',
            'url': '<xsl:value-of select="@url" />',
            'input_src_port':  '<xsl:value-of select="@input_src_port" />',
            'input_dst_port':  '<xsl:value-of select="@input_dst_port" />',
            'output_src_port': '<xsl:value-of select="@output_src_port" />',
            'output_dst_port': '<xsl:value-of select="@output_dst_port" />',
                    <xsl:message terminate="no">
Warning: Post Synapse not fully supported
                    </xsl:message>

            'parameters':{
            <xsl:for-each select="*">
                    '<xsl:value-of select="@name" />':{
                    'dimention':'<xsl:value-of select="@dimension" />',
<xsl:for-each select="*">
                    'input':{'type':'<xsl:value-of select="name()"/>',
                            <xsl:if test="name() = 'FixedValue'">'value':<xsl:value-of select="@value"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'ValueList'">
                                <xsl:message terminate="yes"> Parameter type  <xsl:value-of select="name()"/> used, currently not supported </xsl:message>
                            </xsl:if>
                            <xsl:if test="name() = 'UniformDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'maximum':<xsl:value-of select="@maximum"/>,
                                'minimum':<xsl:value-of select="@minimum"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'NormalDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'mean':<xsl:value-of select="@mean"/>,
                                'variance':<xsl:value-of select="@variance"/>}
                            </xsl:if>
                            <xsl:if test="name() = 'PoissonDistribution'">
                                'seed':<xsl:value-of select="@seed"/>,
                                'mean':<xsl:value-of select="@mean"/>}
                            </xsl:if>
                         </xsl:for-each>
                     }<xsl:choose><xsl:when test="position() != last()">,</xsl:when></xsl:choose>
            </xsl:for-each>
            }
        }
</xsl:template>

<xsl:template match="EX:Simulation">
exp.set_simulation(<xsl:value-of select="@duration"/>,<xsl:if test="EX:EulerIntegration"><xsl:value-of select="EX:EulerIntegration/@dt"/>,'<xsl:value-of select="@preferred_simulator"/>'</xsl:if>
<xsl:if test="EX:RungeKuttaIntegration"><xsl:value-of select="EX:RungeKuttaIntegration/@dt"/>,'<xsl:value-of select="@preferred_simulator"/>',<xsl:value-of select="EX:RungeKuttaIntegration/@order"/></xsl:if>)
</xsl:template>




    <xsl:template match="EX:Lesion">
       exp.add_lesion()
    </xsl:template>

    <xsl:template match="EX:Configuration">
       exp.add_lesion()
    </xsl:template>

</xsl:stylesheet>



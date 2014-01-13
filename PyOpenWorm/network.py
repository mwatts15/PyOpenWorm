# -*- coding: utf-8 -*-
"""
.. class:: Network

This module contains the class that defines the network

"""

import pymysql
from rdflib import Graph
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from rdflib import URIRef, BNode, Literal
import urllib, urllib2
import PyOpenWorm
import networkx as nx
import csv

class Network:

	#def __init__(self):
	
#	def motor(self):
#		return motor
	
	def aneuron(self, name):
		"""
		Get a neuron by name
			
		:param name: Name of a c. elegans neuron
		:returns: Corresponding neuron to the name given
		:rtype: PyOpenWorm.Neuron
		"""
		conn = pymysql.connect(host='my01.winhost.com', port=3306, user='openworm', passwd='openworm', db='mysql_31129_celegans')
	   
		cur = conn.cursor()
	
		#first step, grab all entities and add them to the graph
	
		cur.execute("SELECT DISTINCT ID, Entity FROM tblentity")
	
		n = Namespace("http://openworm.org/entities/")
	
		# print cur.description
	
		g = Graph()
	
		for r in cur.fetchall():
			#first item is a number -- needs to be converted to a string
			first = str(r[0])
			#second item is text 
			second = str(r[1])
		
			# This is the backbone of any RDF graph.  The unique
			# ID for each entity is encoded as a URI and every other piece of 
			# knowledge about that entity is connected via triples to that URI
			# In this case, we connect the common name of that entity to the 
			# root URI via the RDFS label property.
			g.add( (n[first], RDFS.label, Literal(second)) )
	
	
		#second stem, get the relationships between them and add them to the graph
		cur.execute("SELECT DISTINCT EnID1, Relation, EnID2 FROM tblrelationship")
	
		for r in cur.fetchall():
			#print r
			#all items are numbers -- need to be converted to a string
			first = str(r[0])
			second = str(r[1])
			third = str(r[2])
		
			g.add( (n[first], n[second], n[third]) )
	
		cur.close()
		conn.close()
	
		print("going to get results...")
	
		qres = g.query(
		  """SELECT ?predLabel ?objLabel     #we want to get out the labels associated with the predicates and the objects
		   WHERE {
			  ?AVALnode ?p "AVAL".          #we are looking first for the 'AVALnode' that is the anchor of all information about the AVAL neuron
			  ?AVALnode ?predicate ?object .# having identified that node, here we match any predicate and object associated with the AVALnode
			  ?predicate rdfs:label ?predLabel .#for the predicates, look up their plain text label (because otherwise we only have URIs)
			  ?object rdfs:label ?objLabel  #for the object, look up their plain text label.
			} ORDER BY ?predLabel           #sort by the predicate""")       
	
		print("printing results")
	
		print("The graph has " + str(len(g)) + " items in it\n\n")
	
		print "AVAL has the following information stored about it: \n"
		for r in qres.result:
			print str(r[0]), str(r[1])
	
		# when done!
		g.close()
		return PyOpenWorm.Neuron()

	def as_networkx(self):
		"""
		Get the network represented as a `NetworkX <http://networkx.github.io/documentation/latest/>`_ graph
			
		Nodes include neuron name and neuron type.  Edges include kind of synapse and neurotransmitter.
			
		:returns: directed graphs with neurons as verticies and gap junctions and chemical synapses as edges
		:rtype: `NetworkX.DiGraph <http://networkx.github.io/documentation/latest/reference/classes.digraph.html?highlight=digraph#networkx.DiGraph>`_
		"""
		self.worm = nx.DiGraph()
		
		# Neuron table
		csvfile = urllib2.urlopen('https://raw.github.com/openworm/data-viz/master/HivePlots/neurons.csv')
		
		reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in reader:
			neurontype = ""
			# Detects neuron function
			if "sensory" in row[1].lower():
				neurontype += "sensory"
			if "motor" in row[1].lower():
				neurontype += "motor"    
			if "interneuron" in row[1].lower():
				neurontype += "interneuron"
			if len(neurontype) == 0:
				neurontype = "unknown"
				
			if len(row[0]) > 0: # Only saves valid neuron names
				self.worm.add_node(row[0], ntype = neurontype)
		
		# Connectome table
		csvfile = urllib2.urlopen('https://raw.github.com/openworm/data-viz/master/HivePlots/connectome.csv')
		
		reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in reader:
			self.worm.add_edge(row[0], row[1], weight = row[3])
			self.worm[row[0]][row[1]]['synapse'] = row[2]
			self.worm[row[0]][row[1]]['neurotransmitter'] = row[4]
			
		return self.worm
		
	#def sensory(self):
	
	#def inter(self):
	
	#def neuroml(self):
	
	#def rdf(self):
	
	#def networkx(self):
	
	
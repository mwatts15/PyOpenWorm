import rdflib as R

from PyOpenWorm.connection import Connection
from PyOpenWorm.cell import Cell
from PyOpenWorm.neuron import Neuron

from DataTestTemplate import _DataTest


class ConnectionTest(_DataTest):
    def setUp(self):
        _DataTest.setUp(self)
        ns = self.config['rdf.namespace']
        self.trips = [
                (ns['john'], R.RDF['type'], ns['Connection']),
                (ns['john'], Connection.rdf_namespace['pre_cell'], Neuron.rdf_namespace['PVCR']),
                (Neuron.rdf_namespace['PVCR'], Cell.rdf_namespace['name'], R.Literal("PVCR")),
                (Neuron.rdf_namespace['PVCR'], R.RDF['type'], Cell.rdf_type),
                (Neuron.rdf_namespace['PVCR'], R.RDF['type'], Neuron.rdf_type),
                (ns['john'], Connection.rdf_namespace['syntype'], R.Literal("send")),
                (ns['john'], Connection.rdf_namespace['number'], R.Literal('1', datatype=R.XSD.integer)),

                (ns['john'], Connection.rdf_namespace['post_cell'], Neuron.rdf_namespace['AVAL']),
                (Neuron.rdf_namespace['AVAL'], Cell.rdf_namespace['name'], R.Literal("AVAL")),
                (Neuron.rdf_namespace['AVAL'], R.RDF['type'], Cell.rdf_type),
                (Neuron.rdf_namespace['AVAL'], R.RDF['type'], Neuron.rdf_type),

                (ns['luke'], R.RDF['type'], ns['Connection']),
                (ns['luke'], Connection.rdf_namespace['pre_cell'], Neuron.rdf_namespace['PVCL']),
                (Neuron.rdf_namespace['PVCL'], Cell.rdf_namespace['name'], R.Literal("PVCL")),
                (Neuron.rdf_namespace['PVCL'], R.RDF['type'], Cell.rdf_type),
                (Neuron.rdf_namespace['PVCL'], R.RDF['type'], Neuron.rdf_type),
                (ns['luke'], Connection.rdf_namespace['syntype'], R.Literal("send")),
                (ns['luke'], Connection.rdf_namespace['number'], R.Literal('1', datatype=R.XSD.integer)),
                (Neuron.rdf_namespace['AVAR'], Cell.rdf_namespace['name'], R.Literal("AVAR")),
                (Neuron.rdf_namespace['AVAR'], R.RDF['type'], Cell.rdf_type),
                (Neuron.rdf_namespace['AVAR'], R.RDF['type'], Neuron.rdf_type),
                (ns['luke'], Connection.rdf_namespace['post_cell'], Neuron.rdf_namespace['AVAR'])]

    def test_init(self):
        """Initialization with positional parameters"""
        c = Connection('AVAL', 'ADAR', 3, 'send', 'Serotonin')
        self.assertIsInstance(c.pre_cell(), Neuron)
        self.assertIsInstance(c.post_cell(), Neuron)
        self.assertEqual(c.number(), 3)
        self.assertEqual(c.syntype(), 'send')
        self.assertEqual(c.synclass(), 'Serotonin')

    def test_init_number_is_a_number(self):
        with self.assertRaises(Exception):
            Connection(1, 2, "gazillion", 4, 5)

    def test_init_with_neuron_objects(self):
        n1 = Neuron(name="AVAL")
        n2 = Neuron(name="PVCR")
        try:
            Connection(n1, n2)
        except:
            self.fail("Shouldn't fail on Connection init")

    def test_load1(self):
        """ Put the appropriate triples in. Try to load them """
        g = R.Graph()
        self.config['rdf.graph'] = g
        for t in self.trips:
            g.add(t)
        c = Connection(conf=self.config)
        conns = set(c.load())
        self.assertNotEqual(0, len(conns))
        for x in conns:
            self.assertIsInstance(x, Connection)

    def test_load_with_filter(self):
        """ Put the appropriate triples in. Try to load them """
        g = R.Graph()
        self.config['rdf.graph'] = g
        for t in self.trips:
            g.add(t)
        c = Connection(pre_cell="PVCR", conf=self.config)
        g0 = R.Graph()
        for tr in c.full_graph():
            g0.add(tr)
        r = set(c.load())
        self.assertNotEqual(0, len(r))
        for x in r:
            self.assertIsInstance(x, Connection)

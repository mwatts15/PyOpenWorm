from __future__ import print_function

# A consolidation of the data sources for the project
# includes:
# NetworkX!
# RDFlib!
# Other things!
#
# Works like Configure:
# Inherit from the DataUser class to access data of all kinds (listed above)

import sys
import sqlite3
import networkx as nx
import PyOpenWorm
from PyOpenWorm import Configureable, Configure, ConfigValue, BadConf
import hashlib
import csv
from rdflib import URIRef, Literal, Graph, Namespace, ConjunctiveGraph
from rdflib.namespace import RDFS, RDF, NamespaceManager
from datetime import datetime as DT
import datetime
import transaction
import os
import traceback
import logging
from itertools import islice, tee

from urllib2 import URLError
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from socket import error as SocketError
import errno

L = logging.Logger("PyOpenWorm.data")

__all__ = [
    "Data",
    "DataUser",
    "RDFSource",
    "SerializationSource",
    "TrixSource",
    "SPARQLSource",
    "SleepyCatSource",
    "DefaultSource",
    "ZODBSource"]


class _B(ConfigValue):

    def __init__(self, f):
        self.v = False
        self.f = f

    def get(self):
        if not self.v:
            self.v = self.f()

        return self.v

    def invalidate(self):
        self.v = False

ZERO = datetime.timedelta(0)


class _UTC(datetime.tzinfo):

    """UTC"""

    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO
utc = _UTC()

propertyTypes = {"send": 'http://openworm.org/entities/356',
                 "Neuropeptide": 'http://openworm.org/entities/354',
                 "Receptor": 'http://openworm.org/entities/361',
                 "is a": 'http://openworm.org/entities/1515',
                 "neuromuscular junction": 'http://openworm.org/entities/1516',
                 "Innexin": 'http://openworm.org/entities/355',
                 "Neurotransmitter": 'http://openworm.org/entities/313',
                 "gap junction": 'http://openworm.org/entities/357'}


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    while True:
        l = []
        try:
            for x in args:
                l.append(next(x))
        except:
            pass
        yield l
        if len(l) < n:
            break


class DataUser(Configureable):

    """ A convenience wrapper for users of the database

    Classes which use the database should inherit from DataUser.
    """

    def __init__(self, **kwargs):
        super(DataUser, self).__init__(**kwargs)
        if not isinstance(self.conf, Data):
            raise BadConf("Not a Data instance: " + str(self.conf))

    @property
    def base_namespace(self):
        return self.conf['rdf.namespace']

    @base_namespace.setter
    def base_namespace_set(self, value):
        self.conf['rdf.namespace'] = value

    @property
    def rdf(self):
        return self.conf['rdf.graph']

    @property
    def namespace_manager(self):
        return self.conf['rdf.namespace_manager']

    @rdf.setter
    def rdf(self, value):
        self.conf['rdf.graph'] = value

    def _remove_from_store(self, g):
        statement_count = 1000
        if self.conf['rdf.store'] == 'SPARQLUpdateStore':
            statement_count = self.conf.get(
                'rdf.upload_block_statement_count',
                default=statement_count)

        for group in grouper(g, statement_count):
            temp_graph = Graph()
            for x in group:
                if x is not None:
                    temp_graph.add(x)
                else:
                    break
            s = " DELETE DATA {" + temp_graph.serialize(format="nt") + " } "
            L.debug("deleting. s = " + s)
            self.conf['rdf.graph'].update(s)

    def _add_to_store(self, g, graph_name=None):
        if self.conf['rdf.store'] == 'SPARQLUpdateStore':
            # XXX With Sesame, for instance, it is probably faster to do a PUT over
            # the endpoint's rest interface. Just need to do it for some common
            # endpoints
            g = iter(g)
            conf = 'rdf.upload_block_statement_count'
            default = 50
            if conf not in self.conf:
                L.warning(
                    "No {conf} in configuration. Defaulting to {default} triples at a time".format(
                        locals()))

            statement_count = self.conf.get(conf, default=default)

            # default count arrived at experimentally with OpenRDF on an AWS
            # t2.micro instance with 1 GB of memory. On the PyOpenWorm side,
            # a socket timeout of 30s was used and the upload statement was
            # sent 'raw' (without URL encoding). Beyond that, the server ran
            # out of memory.
            max_statement_count = self.conf.get('rdf.upload_block_statement_count.max', 65536)
            method = self.conf.get('rdf.upload_block_statement_count.method', 'constant')
            method_conf = self.conf.get('rdf.upload_block_statement_count.method_conf', {})
            sizer_class = statement_count_methods[method]['class']
            slice_sizer = sizer_class(self.conf)
            next_size = slice_sizer.init_size()

            graphs = [g]
            group = None

            while True:
                print(graphs)
                graphs.append(set(islice(graphs[len(graphs)-1], next_size)))

                try:
                    print("About to add {} statements".format(next_size))
                    if self._add_to_storeb(graphs[len(graphs)-1], graph_name, next_size) == 0:
                        break
                    next_size = slice_sizer.success()
                except _SliceSizerException as e:
                    break
                except (URLError, EndPointInternalError) as e:
                    if ("time" in str(e)) or isinstance(e, EndPointInternalError):
                        next_size = slice_sizer.failure()
                        graphs.append(graphs[len(graphs)-1])
                        # TODO: Close and re-open the connection
                    else:
                        traceback.print_exc()
                        break
                except SocketError as e:
                    if e.errno != errno.ECONNRESET:
                        raise
                    # TODO: Close and re-open the connection
                    # connection reset errors are (hopefully) transient, so we just
                    # retry with a new connection
                    continue
                except Exception:
                    traceback.print_exc()
                    break

                graphs.pop()

                if next_size > max_statement_count:
                    slice_sizer.set_size(max_statement_count)
                    next_size = max_statement_count
        else:
            gr = self.conf['rdf.graph']
            for x in g:
                gr.add(x)

        if self.conf['rdf.source'] == 'ZODB':
            # Commit the current transaction
            transaction.commit()
            # Fire off a new one
            transaction.begin()

    def _add_to_storeb(self, g, graph_name, n_triples):
        gs = self._serialize_upload_triples(g, n_triples)
        c = len(gs)
        if c > 0:
            if graph_name is not None:
                s = "INSERT DATA { GRAPH " + graph_name.n3() + " {" + gs + " } } "
            else:
                s = "INSERT DATA { " + gs + " } "
            L.debug("update query = " + s)
            self.conf['rdf.graph'].update(s)
            self.conf['rdf.graph'].commit()
        return c
        # infer from the added statements
        # self.infer()

    def _serialize_upload_triples(self, g, n_triples, initial_buffer_size=None):
        """ Serializes triples made of only URIRef and Literal for SPARQL "INSERT DATA" """
        if initial_buffer_size is None:
            initial_buffer_size = 24000 # TODO: Allow to configure

        ba = bytearray(initial_buffer_size)
        stmt_start = 0
        buf_index = 0
        stmt_idx = 0
        while True:
            try:
                for (i, t) in enumerate(g):
                    stmt_idx = i
                    stmt_start = buf_index
                    s = t[0].encode('UTF-8')
                    sl = len(s)
                    p = t[1].encode('UTF-8')
                    pl = len(p)
                    o = t[2].encode('UTF-8')
                    ol = len(o)

                    ba[buf_index:buf_index+1] = b'<'
                    buf_index += 1
                    ba[buf_index:buf_index + sl] = s
                    buf_index += sl
                    ba[buf_index:buf_index + 2] = b'><'
                    buf_index += 2
                    ba[buf_index:buf_index + pl] = p
                    buf_index += pl
                    ba[buf_index:buf_index + 1] = b'>'
                    buf_index += 1
                    if isinstance(t[2], Literal):
                        ba[buf_index:buf_index + 1] = b'"'
                        buf_index += 1
                        ba[buf_index:buf_index + ol] = o
                        buf_index += ol
                        ba[buf_index:buf_index + 1] = b'"'
                        buf_index += 1
                        lang = t[2].language
                        dt = t[2].datatype
                        if lang:
                            enc = lang.encode('UTF-8')
                            ll = len(enc)
                            ba[buf_index:buf_index + 1] = b'@'
                            buf_index += 1
                            ba[buf_index:buf_index + ll] = enc
                            buf_index += ll
                        elif dt:
                            enc = dt.encode('UTF-8')
                            ll = len(enc)
                            ba[buf_index:buf_index + 3] = b'^^<'
                            buf_index += 3
                            ba[buf_index:buf_index + ll] = enc
                            buf_index += ll
                            ba[buf_index:buf_index + 2] = b'>.'
                            buf_index += 2
                        else:
                            ba[buf_index:buf_index + 1] = b'.'
                            buf_index += 1
                    else:
                        ba[buf_index:buf_index + 1] = b'<'
                        buf_index += 1
                        ba[buf_index:buf_index + ol] = o
                        buf_index += ol
                        ba[buf_index:buf_index + 2] = b'>.'
                        buf_index += 2
            except IndexError:
                buf_index = stmt_start
                # estimate how much more we need based on how much we've
                # used
                estimate_on_space_needed = (buf_index // stmt_idx) * (n_triples - stmt_idx)
                ba.extend(bytearray(estimate_on_space_needed))
                continue
            break
        s = ba[0:buf_index].decode('UTF-8')
        #print(s)
        return s


    def infer(self):
        """ Fire FuXi rule engine to infer triples """

        from FuXi.Rete.RuleStore import SetupRuleStore
        from FuXi.Rete.Util import generateTokenSet
        from FuXi.Horn.HornRules import HornFromN3
        # fetch the derived object's graph
        semnet = self.rdf
        rule_store, rule_graph, network = SetupRuleStore(makeNetwork=True)
        closureDeltaGraph = Graph()
        network.inferredFacts = closureDeltaGraph
        # build a network of rules
        for rule in HornFromN3('testrules.n3'):
            network.buildNetworkFromClause(rule)
        # apply rules to original facts to infer new facts
        network.feedFactsToAdd(generateTokenSet(semnet))
        # combine original facts with inferred facts
        for x in closureDeltaGraph:
            self.rdf.add(x)

    def add_reference(self, g, reference_iri):
        """
        Add a citation to a set of statements in the database

        :param triples: A set of triples to annotate
        """
        new_statements = Graph()
        ns = self.conf['rdf.namespace']
        for statement in g:
            statement_node = self._reify(new_statements, statement)
            new_statements.add(
                (URIRef(reference_iri),
                 ns['asserts'],
                 statement_node))

        self.add_statements(g + new_statements)

    # def _add_unannotated_statements(self, graph):
    # A UTC class.

    def retract_statements(self, graph):
        """
        Remove a set of statements from the database.

        :param graph: An iterable of triples
        """
        self._remove_from_store_by_query(graph)

    def _remove_from_store_by_query(self, q):
        s = " DELETE WHERE {" + q + " } "
        L.debug("deleting. s = " + s)
        self.conf['rdf.graph'].update(s)

    def add_statements(self, graph):
        """
        Add a set of statements to the database.
        Annotates the addition with uploader name, etc

        :param graph: An iterable of triples
        """
        self._add_to_store(graph)

    def _reify(self, g, s):
        """
        Add a statement object to g that binds to s
        """
        n = self.conf['new_graph_uri'](s)
        g.add((n, RDF['type'], RDF['Statement']))
        g.add((n, RDF['subject'], s[0]))
        g.add((n, RDF['predicate'], s[1]))
        g.add((n, RDF['object'], s[2]))
        return n


class _SliceSizerException(Exception):
    pass


class _SliceSizer(object):
    """ Provides sizes for the number of statements to send at once through
    a series of successes and failures in sending
    """
    def __init__(self, conf):
        """ Takes a Configuration object to parameterize the sizer's behavior """
        self.conf = conf
        self.count = self.conf.get('rdf.upload_block_statement_count', 1)
        if self.count < 1:
            raise ValueError('Configuration value ''rdf.upload_block_statement_count'' must be greater than 0')

        self.sizer_conf = self.conf.get('rdf.upload_block_statement_count.method_conf', {})
        self.max_retries = self.sizer_conf.get('max_retries', 10)

    def init_size(self):
        """ Get the initial size for this sizer """
        raise NotImplementedError()

    def success(self):
        """ Get the size for this sizer after a success """
        raise NotImplementedError()

    def failure(self):
        """ Get the size for this sizer after a failure """
        raise NotImplementedError()

    def set_size(self):
        raise NotImplementedError()


class _ConstantSizer(_SliceSizer):

    def __init__(self, conf):
        super(_ConstantSizer, self).__init__(conf)
        self.retries = 0
        self.size = self.count

    def init_size(self):
        return self.size

    def success(self):
        return self.size

    def failure(self):
        if self.retries > self.max_retries:
            raise _SliceSizerException("Max retries reached")
        self.retries += 1
        return self.size

    def set_size(self, size):
        self.size = size


class _BinarySearchSizerMixin(object):

    def __init__(self, conf):
        super(_BinarySearchSizerMixin, self).__init__(conf)
        self.do_binary_search = False
        self.binsearchsizer = None
        self.last_success_size = 0
        self.last_failure_size = 0
        self.optimized = False
        self.retries = 0

    def success(self):
        self.last_success_size = self.size
        if not self.optimized:
            if self.do_binary_search:
                self.size = self.binsearchsizer.success()
            else:
                self.size = super(_BinarySearchSizerMixin, self).success()

            if self.last_success_size == self.size:
                self.optimized = True
        return self.size

    def _reset(self):
        # Reset ourselves
        self.retries = 0
        self.do_binary_search = False
        self.binsearchsizer = None
        self.optimized = False

    def failure(self):
        self.last_failure_size = self.size
        if self.optimized:
            self.retries += 1
            if self.retries > self.max_retries:
                self.size = self.size // 2
                if self.size == 0:
                    raise _SliceSizerException("Couldn't find a size on which to transmit")
                self._reset()
        elif self.do_binary_search:
            self.size = self.binsearchsizer.failure()
            if self.size == self.last_failure_size:
                self.retries += 1
            else:
                self.retries = 0
            if self.retries > self.max_retries:
                raise _SliceSizerException('Max retries reached')
        else:
            self.do_binary_search = True
            self.binsearchsizer = _BinarySearchSizer(self.conf, self.last_success_size, self.size)
            self.size = self.binsearchsizer.failure()
        return self.size

    def set_size(self, size):
        self._reset()
        self.size = size

class _LinearSizer(_SliceSizer):

    def __init__(self, conf):
        super(_LinearSizer, self).__init__(conf)
        self.size = self.count

    def init_size(self):
        return self.count

    def success(self):
        self.size += self.count
        return self.size

    def set_size(self, size):
        self._reset()
        self.size = size


class _GeometricSizer(_SliceSizer):

    def __init__(self, conf):
        super(_GeometricSizer, self).__init__(conf)
        self.size = self.count

    def init_size(self):
        return self.count

    def success(self):
        self.size *= 2
        return self.size

    def set_size(self, size):
        self._reset()
        self.size = size


class _LinearSizerWithBinarySearch(_BinarySearchSizerMixin, _LinearSizer):
    pass


class _GeometricSizerWithBinarySearch(_BinarySearchSizerMixin, _GeometricSizer):
    pass


class _BinarySearchSizer(_SliceSizer):
    def __init__(self, conf, low, hi):
        super(_BinarySearchSizer, self).__init__(conf)
        self.low = low
        self.init = low
        self.size = hi
        self.hi = hi

    def init_size(self):
        return self.init

    def success(self):
        self.low = self.size
        self.size = (self.low + self.hi) // 2
        return self.size

    def failure(self):
        self.hi = self.size
        self.size = (self.low + self.hi) // 2
        return self.size


statement_count_methods = {
    'linear': {
        'desc': """Linear increase

        The increments are by rdf.upload_block_statement_count. For a failure,
        the algorithm begins a binary search between the last success and the
        last failure to find the maximum count that succeeds. Subsequent
        failures at that maximum cause a fixed number of retries followed by
        exponential back-off.
        """,
        'class': _LinearSizerWithBinarySearch},
    'geometric': {
        'desc': """Geometric increase (doubling)

        For a failure, the algorithm begins a binary search between the last
        success and the last failure to find the maximum count that succeeds.
        Subsequent failures at that maximum cause a fixed number of retries
        followed by exponential back-off.
        """,
        'class': _GeometricSizerWithBinarySearch},
    'constant': {
        'desc': """Uses rdf.upload_block_statement_count with no variation.

        After a fixed number of retries, the upload simply fails.
        """,
        'func': _ConstantSizer}
}


class Data(Configure, Configureable):

    """
    Provides configuration for access to the database.

    Usally doesn't need to be accessed directly
    """

    def __init__(self, conf=False):
        Configure.__init__(self)
        Configureable.__init__(self)
        # We copy over all of the configuration that we were given
        if conf:
            self.copy(conf)
        else:
            self.copy(Configureable.conf)
        self.namespace = Namespace("http://openworm.org/entities/")
        self.molecule_namespace = Namespace(
            "http://openworm.org/entities/molecules/")
        self['nx'] = _B(self._init_networkX)
        self['rdf.namespace'] = self.namespace
        self['molecule_name'] = self._molecule_hash
        self['new_graph_uri'] = self._molecule_hash

    @classmethod
    def open(cls, file_name):
        """ Open a file storing configuration in a JSON format """
        Configureable.conf = Configure.open(file_name)
        return cls()

    def openDatabase(self):
        """ Open a the configured database """
        self._init_rdf_graph()
        L.debug("opening " + str(self.source))
        self.source.open()
        nm = NamespaceManager(self['rdf.graph'])
        self['rdf.namespace_manager'] = nm
        self['rdf.graph'].namespace_manager = nm

        nm.bind("", self['rdf.namespace'])

    def closeDatabase(self):
        """ Close a the configured database """
        self.source.close()

    def _init_rdf_graph(self):
        # Set these in case they were left out
        c = self.conf
        self['rdf.source'] = c['rdf.source'] = c.get('rdf.source', 'default')
        self['rdf.store'] = c['rdf.store'] = c.get('rdf.store', 'default')
        self['rdf.store_conf'] = c['rdf.store_conf'] = c.get(
            'rdf.store_conf',
            'default')

        # XXX:The conf=self can probably be removed
        self.sources = {'sqlite': SQLiteSource,
                        'sparql_endpoint': SPARQLSource,
                        'sleepycat': SleepyCatSource,
                        'default': DefaultSource,
                        'trix': TrixSource,
                        'serialization': SerializationSource,
                        'zodb': ZODBSource
                        }
        i = self.sources[self['rdf.source'].lower()]()
        self.source = i
        self.link('semantic_net_new', 'semantic_net', 'rdf.graph')
        self['rdf.graph'] = i
        return i

    def _molecule_hash(self, data):
        return URIRef(
            self.molecule_namespace[
                hashlib.sha224(
                    str(data)).hexdigest()])

    def _init_networkX(self):
        g = nx.DiGraph()

        # Neuron table
        csvfile = open(self.conf['neuronscsv'])

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

            if len(row[0]) > 0:  # Only saves valid neuron names
                g.add_node(row[0], ntype=neurontype)

        # Connectome table
        csvfile = open(self.conf['connectomecsv'])

        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            g.add_edge(row[0], row[1], weight=row[3])
            g[row[0]][row[1]]['synapse'] = row[2]
            g[row[0]][row[1]]['neurotransmitter'] = row[4]
        return g


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


class RDFSource(Configureable, PyOpenWorm.ConfigValue):

    """ Base class for data sources.

    Alternative sources should dervie from this class
    """
    i = 0

    def __init__(self, **kwargs):
        if self.i == 1:
            raise Exception(self.i)
        self.i += 1
        Configureable.__init__(self, **kwargs)
        self.graph = False

    def get(self):
        if self.graph == False:
            raise Exception(
                "Must call openDatabase on Data object before using the database")
        return self.graph

    def close(self):
        if self.graph == False:
            return
        self.graph.close()
        self.graph = False

    def open(self):
        """ Called on ``PyOpenWorm.connect()`` to set up and return the rdflib graph.
        Must be overridden by sub-classes.
        """
        raise NotImplementedError()


class SerializationSource(RDFSource):

    """ Reads from an RDF serialization or, if the configured database is more recent, then from that.

        The database store is configured with::

            "rdf.source" = "serialization"
            "rdf.store" = <your rdflib store name here>
            "rdf.serialization" = <your RDF serialization>
            "rdf.serialization_format" = <your rdflib serialization format used>
            "rdf.store_conf" = <your rdflib store configuration here>

    """

    def open(self):
        if not self.graph:
            self.graph = True
            import glob
            # Check the ages of the files. Read the more recent one.
            g0 = ConjunctiveGraph(store=self.conf['rdf.store'], identifier=self.conf['rdf.graph_id'])
            database_store = self.conf['rdf.store_conf']
            source_file = self.conf['rdf.serialization']
            file_format = self.conf['rdf.serialization_format']
            # store_time only works for stores that are on the local
            # machine.
            try:
                store_time = modification_date(database_store)
                # If the store is newer than the serialization
                # get the newest file in the store
                for x in glob.glob(database_store + "/*"):
                    mod = modification_date(x)
                    if store_time < mod:
                        store_time = mod
            except:
                store_time = DT.min

            trix_time = modification_date(source_file)

            g0.open(database_store, create=True)

            if store_time > trix_time:
                # just use the store
                pass
            else:
                # delete the database and read in the new one
                # read in the serialized format
                g0.parse(source_file, format=file_format)

            self.graph = g0

        return self.graph


class TrixSource(SerializationSource):

    """ A SerializationSource specialized for TriX

        The database store is configured with::

            "rdf.source" = "trix"
            "rdf.trix_location" = <location of the TriX file>
            "rdf.store" = <your rdflib store name here>
            "rdf.store_conf" = <your rdflib store configuration here>

    """

    def __init__(self, **kwargs):
        SerializationSource.__init__(self, **kwargs)
        h = self.conf.get('trix_location', 'UNSET')
        self.conf.link('rdf.serialization', 'trix_location')
        self.conf['rdf.serialization'] = h
        self.conf['rdf.serialization_format'] = 'trix'


def _rdf_literal_to_gp(x):
    return x.n3()


def _triples_to_bgp(trips):
    # XXX: Collisions could result between the variable names of different
    # objects
    g = " .\n".join(" ".join(_rdf_literal_to_gp(x) for x in y) for y in trips)
    return g


class SPARQLSource(RDFSource):

    """ Reads from and queries against a remote data store

        ::

            "rdf.source" = "sparql_endpoint"
            "rdf.store_conf" = <your SPARQL endpoint here>
    """

    def open(self):
        # XXX: If we have a source that's read only, should we need to set the
        # store separately??
        g0 = Graph('SPARQLUpdateStore', identifier=self.conf['rdf.graph_id'])
        conf = None
        store_conf = self.conf['rdf.store_conf']
        timeout_conf = 'rdf.sparql_endpoint.upload_timeout'
        if timeout_conf in self.conf:
            g0.store.setTimeout(self.conf[timeout_conf])
        g0.store.postAsEncoded = False

        if isinstance(store_conf, list):
            conf = tuple(store_conf)
        elif isinstance(store_conf, str):
            conf = (store_conf,)
        else:
            raise Exception("The rdf.store_conf for a SPARQLSource "
                            "should be of the form "
                            "'[<query_endpoint>, <update_endpoint>]' "
                            "'[<sparql_endpoint>]' "
                            "or '<sparql_endpoint>'")

        g0.open(conf)
        self.conf['rdf.store'] = 'SPARQLUpdateStore'
        self.graph = g0
        return self.graph

    def __repr__(self):
        return "SPARQLSource(" + str(self.conf['rdf.store_conf']) + ")"

    def __str__(self):
        return repr(self)


class SleepyCatSource(RDFSource):

    """ Reads from and queries against a local Sleepycat database

        The database can be configured like::

            "rdf.source" = "Sleepycat"
            "rdf.store_conf" = <your database location here>
    """

    def open(self):
        # XXX: If we have a source that's read only, should we need to set the
        # store separately??
        g0 = ConjunctiveGraph('Sleepycat', identifier=self.conf['rdf.graph_id'])
        self.conf['rdf.store'] = 'Sleepycat'
        g0.open(self.conf['rdf.store_conf'], create=True)
        self.graph = g0
        L.debug("Opened SleepyCatSource")


class SQLiteSource(RDFSource):

    """ Reads from and queries against a SQLite database

    See see the SQLite database :file:`db/celegans.db` for the format

    The database store is configured with::

        "rdf.source" = "Sleepycat"
        "sqldb" = "/home/USER/openworm/PyOpenWorm/db/celegans.db",
        "rdf.store" = <your rdflib store name here>
        "rdf.store_conf" = <your rdflib store configuration here>

    Leaving ``rdf.store`` unconfigured simply gives an in-memory data store.
    """

    def open(self):
        conn = sqlite3.connect(self.conf['sqldb'])
        cur = conn.cursor()

        # first step, grab all entities and add them to the graph
        n = self.conf['rdf.namespace']

        cur.execute("SELECT DISTINCT ID, Entity FROM tblentity")
        g0 = ConjunctiveGraph(self.conf['rdf.store'], identifier=self.conf['rdf.graph_id'])
        g0.open(self.conf['rdf.store_conf'], create=True)

        for r in cur.fetchall():
            # first item is a number -- needs to be converted to a string
            first = str(r[0])
            # second item is text
            second = str(r[1])

            # This is the backbone of any RDF graph.  The unique
            # ID for each entity is encoded as a URI and every other piece of
            # knowledge about that entity is connected via triples to that URI
            # In this case, we connect the common name of that entity to the
            # root URI via the RDFS label property.
            g0.add((n[first], RDFS.label, Literal(second)))

        # second step, get the relationships between them and add them to the
        # graph
        cur.execute(
            "SELECT DISTINCT EnID1, Relation, EnID2, Citations FROM tblrelationship")

        gi = ''

        i = 0
        for r in cur.fetchall():
            # all items are numbers -- need to be converted to a string
            first = str(r[0])
            second = str(r[1])
            third = str(r[2])
            prov = str(r[3])

            ui = self.conf['molecule_name'](prov)
            gi = Graph(g0.store, ui)

            gi.add((n[first], n[second], n[third]))

            g0.add([ui, RDFS.label, Literal(str(i))])
            if (prov != ''):
                g0.add([ui, n[u'text_reference'], Literal(prov)])

            i = i + 1

        cur.close()
        conn.close()
        self.graph = g0


class DefaultSource(RDFSource):

    """ Reads from and queries against a configured database.

        The default configuration.

        The database store is configured with::

            "rdf.source" = "default"
            "rdf.store" = <your rdflib store name here>
            "rdf.store_conf" = <your rdflib store configuration here>

        Leaving unconfigured simply gives an in-memory data store.
    """

    def open(self):
        self.graph = ConjunctiveGraph(self.conf['rdf.store'], identifier=self.conf['rdf.graph_id'])
        self.graph.open(self.conf['rdf.store_conf'], create=True)


class ZODBSource(RDFSource):

    """ Reads from and queries against a configured Zope Object Database.

        If the configured database does not exist, it is created.

        The database store is configured with::

            "rdf.source" = "ZODB"
            "rdf.store_conf" = <location of your ZODB database>

        Leaving unconfigured simply gives an in-memory data store.
    """

    def __init__(self, *args, **kwargs):
        RDFSource.__init__(self, *args, **kwargs)
        self.conf['rdf.store'] = "ZODB"

    def open(self):
        import ZODB
        from ZODB.FileStorage import FileStorage
        self.path = self.conf['rdf.store_conf']
        openstr = os.path.abspath(self.path)

        fs = FileStorage(openstr)
        self.zdb = ZODB.DB(fs)
        self.conn = self.zdb.open()
        root = self.conn.root()
        if 'rdflib' not in root:
            root['rdflib'] = ConjunctiveGraph('ZODB', identifier=self.conf['rdf.graph_id'])
        self.graph = root['rdflib']
        try:
            transaction.commit()
        except Exception:
            # catch commit exception and close db.
            # otherwise db would stay open and follow up tests
            # will detect the db in error state
            L.warning('Forced to abort transaction on ZODB store opening')
            traceback.print_exc()
            transaction.abort()
        transaction.begin()
        self.graph.open(self.path)


    def close(self):
        if self.graph == False:
            return

        self.graph.close()

        try:
            transaction.commit()
        except Exception:
            # catch commit exception and close db.
            # otherwise db would stay open and follow up tests
            # will detect the db in error state
            traceback.print_exc()
            L.warning('Forced to abort transaction on ZODB store closing')
            transaction.abort()
        self.conn.close()
        self.zdb.close()
        self.graph = False

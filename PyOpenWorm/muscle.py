import PyOpenWorm as P
from .cell import Cell
from .connection import Connection
from .neuron import Neuron

class Muscle(Cell):
    """A single muscle cell.

    See what neurons innervate a muscle:

    Example::

        >>> mdr21 = P.Muscle('MDR21')
        >>> innervates_mdr21 = mdr21.innervatedBy()
        >>> len(innervates_mdr21)
        4

    Attributes
    ----------
    neurons : ObjectProperty
        Neurons synapsing with this muscle
    receptors : DatatypeProperty
        Get a list of receptors for this muscle if called with no arguments,
        or state that this muscle has the given receptor type if called with
        an argument
    """

    def __init__(self, name=False, **kwargs):
        super(Muscle,self).__init__(name=name, **kwargs)
        self.innervatedBy = Muscle.ObjectProperty("neurons", owner=self, value_type=P.Neuron, multiple=True)
        Muscle.DatatypeProperty("receptors", owner=self, multiple=True)
        self.attach_property_n(InnervatedByProperty)

class InnervatedByProperty(object):
    multiple = True
    linkName = "innervatedBy"

    def __init__(self, conf, owner, resolver):
        self.conf = conf
        self.owner = owner
        #self.link = owner.rdf_namespace[self.linkName]
        self._conns = list()
        # XXX: should the resolver here be made
        # use of?

    def hasValue(self):
        return len(self._conns) > 0

    def has_defined_value(self):
        for x in self._conns:
            if x.defined:
                return True
        return False

    def set(self, v, **kwargs):
        if not isinstance(v, P.Neuron):
            raise Exception("Muscles can only be innervated by neurons")

        conn = Connection(pre_cell=v, post_cell=self.owner, **kwargs)
        conn.termination('muscle')
        self._conns.append(conn)
        # We already pay the full cost of the connection object so we
        # don't bother to make an intermediate object like Rel
        return conn

    def relationships(self):
        return self._conns

    @property
    def defined_values(self):
        return tuple(x.pre_cell.defined_values[0] for x in self._conns if x.defined)

    @property
    def values(self):
        return tuple(x.pre_cell.values[0] for x in self._conns)

    @property
    def rdf(self):
        return self.conf['rdf.graph']

    def get(self, **kwargs):
        n = Neuron()
        #print('muscle', self.owner)
        c = P.Connection(pre_cell=n, post_cell=self.owner, **kwargs)
        res = set(n.load())
        c.post_cell.unset(self.owner)
        return res

    def count(self):
        n = Neuron()
        c = P.Connection(pre_cell=n, post_cell=self.owner, **kwargs)
        res = n.count()
        c.post_cell.unset(self.owner)
        return res

    def unset(self, v):
        for c in self._conns:
            if c.pre_cell() == v:
                c.post_cell.unset(self.owner)

    def __call__(self, *args, **kwargs):
        """ If arguments are passed to the ``Property``, its ``set`` method
        is called. Otherwise, if the object has values set on it, then its
        ``defined_values`` are returned. If no arguments are passed and no
        values have been set, then the ``get`` method is called. If the
        ``multiple`` member for the ``Property`` is set to ``True``, then a
        Python set containing the values is returned. Otherwise, a single bare
        value is returned.
        """

        if len(args) > 0 or len(kwargs) > 0:
            return self.set(*args, **kwargs)
        else:
            if self.has_defined_value():
                return self.defined_values
            else:
                r = self.get()
                return set(r)

    def one(self):
        l = list(self.get())
        if len(l) > 0:
            return l[0]
        else:
            return None


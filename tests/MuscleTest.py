from __future__ import print_function

from PyOpenWorm.muscle import Muscle
from PyOpenWorm.neuron import Neuron
from PyOpenWorm.connection import SynapseType

from DataTestTemplate import _DataTest


class MuscleTest(_DataTest):

    def test_muscle(self):
        self.assertTrue(isinstance(Muscle(name='MDL08'), Muscle))

    def test_innervatedBy(self):
        m = Muscle('MDL08')
        n = Neuron('some neuron')
        m.innervatedBy(n, syntype='send')
        m.save()
        v = Muscle(name='MDL08')
        self.assertIn(n, list(v.innervatedBy()))

    def test_innervatedBy_count(self):
        m = Muscle('MDL08')
        m.innervatedBy(Neuron('some neuron'), syntype=SynapseType.Chemical)
        m.innervatedBy(Neuron('some other neuron'), syntype=SynapseType.Chemical)
        m.save()
        v = Muscle(name='MDL08')
        self.assertEqual(2, v.innervatedBy.count(syntype=SynapseType.Chemical))

    def test_innervatedBy_count_with_args(self):
        m = Muscle('MDL08')
        m.innervatedBy(Neuron('some neuron'), syntype=SynapseType.Chemical)
        m.innervatedBy(Neuron('some other neuron'), syntype=SynapseType.Chemical)
        m.innervatedBy(Neuron('this neuron here'), syntype=SynapseType.GapJunction)
        m.save()
        v = Muscle(name='MDL08')
        self.assertEqual(2, v.innervatedBy.count(syntype=SynapseType.Chemical))

    def test_muscle_neurons(self):
        """ Should be the same as innervatedBy """
        m = Muscle(name='MDL08')
        neu = Neuron(name="tnnetenba")
        m.neurons(neu)
        m.save()

        m = Muscle(name='MDL08')
        self.assertIn(Neuron('tnnetenba'), list(m.neurons()))

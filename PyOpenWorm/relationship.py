from .dataObject import DataObject

class Relationship(DataObject):
    """ A Relationship is typically related to a property and is an object that
        one points to for talking about the property relationship.

        For SimpleProperty objects, this acts like a RDF Reified triple.
        """
    def __init__(self, subject=None, property=None, object=None, **kwargs):
        super(Relationship, self).__init__(**kwargs)
        self.ObjectProperty('subject', owner=self)
        self.ObjectProperty('property', owner=self)
        self.UnionProperty('object', owner=self)

        if subject is not None:
            self.subject(subject)

        if property is not None:
            self.property(property)

        if object is not None:
            self.object(object)

    def _ident_data(self):
        return [self.subject.defined_values,
                self.property.defined_values,
                self.object.defined_values]

    def defined(self):
        if super(Relationship, self).defined:
            return True
        else:
            for p in self._ident_data():
                if len(p) == 0:
                    return False
            return True

    def identifier(self):
        if super(Relationship, self).defined:
            return super(Relationship, self).identifier()
        else:
            data = self._ident_data()
            data = tuple(x[0].identifier().n3() for x in data)
            data = "".join(data)
            return self.make_identifier(data)

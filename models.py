from neomodel import StructuredNode, StructuredRel, StringProperty, IntegerProperty, FloatProperty, BooleanProperty, DateTimeProperty, UniqueIdProperty, RelationshipTo

# Generated Models

class HasRelationWithPropRel(StructuredRel):
    someProp = StringProperty

class FirstLabel(StructuredNode):
    uid = StringProperty(unique_index=True)
    someNumber = IntegerProperty
    someFloat = FloatProperty
    aBool = BooleanProperty
    dob = DateTimeProperty(index=True)
    has_some_relation = RelationshipTo('SecondLabel', 'HAS_SOME_RELATION')
    has_relation_with_prop = RelationshipTo('SecondLabel', 'HAS_RELATION_WITH_PROP', model=HasRelationWithPropRel)

    def to_dict(self):
        props = {}
        for prop_name in self.__all_properties__:
            props[prop_name] = getattr(self, prop_name)
        return props

class SecondLabel(StructuredNode):
    uid = UniqueIdProperty()
    self_relation = RelationshipTo('SecondLabel', 'SELF_RELATION')

    def to_dict(self):
        props = {}
        for prop_name in self.__all_properties__:
            props[prop_name] = getattr(self, prop_name)
        return props

class NodeWithNoRelation(StructuredNode):
    someId = StringProperty(unique_index=True)

    def to_dict(self):
        props = {}
        for prop_name in self.__all_properties__:
            props[prop_name] = getattr(self, prop_name)
        return props

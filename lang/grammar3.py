"""
Python grammar and typing system
"""
import ast
import inspect
from collections import OrderedDict, defaultdict
import logging

from lang.astnode import ASTNode
from lang.util import typename


class Grammar(object):
    def __init__(self, rules):
        """
        instantiate a grammar with a set of production rules of type Rule
        """
        self.rules = rules
        self.rule_index = defaultdict(list)
        self.rule_to_id = OrderedDict()

        node_types = set()
        lhs_nodes = set()
        rhs_nodes = set()
        for rule in self.rules:
            self.rule_index[rule.parent].append(rule)

            # we also store all unique node types
            for node in rule.nodes:
                node_types.add(typename(node.type))

            lhs_nodes.add(rule.parent)
            for child in rule.children:
                rhs_nodes.add(child.as_type_node)

        root_node = lhs_nodes - rhs_nodes
        assert len(root_node) == 1
        self.root_node = next(iter(root_node))

        self.terminal_nodes = rhs_nodes - lhs_nodes
        self.terminal_types = set([n.type for n in self.terminal_nodes])

        self.node_type_to_id = OrderedDict()
        for i, type in enumerate(node_types, start=0):
            self.node_type_to_id[type] = i

        for gid, rule in enumerate(rules, start=0):
            self.rule_to_id[rule] = gid

        self.id_to_rule = OrderedDict((v, k) for (k, v) in self.rule_to_id.items())

        logging.info('num. rules: %d', len(self.rules))
        logging.info('num. types: %d', len(self.node_type_to_id))
        logging.info('root: %s', self.root_node)
        logging.info('terminals: %s', ', '.join(repr(n) for n in self.terminal_nodes))

    def __iter__(self):
        return self.rules.__iter__()

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, lhs):
        key_node = ASTNode(lhs.type, None)  # Rules are indexed by types only
        if key_node in self.rule_index:
            return self.rule_index[key_node]
        else:
            KeyError('key=%s' % key_node)

    def get_node_type_id(self, node):
        if isinstance(node, ASTNode):
            type_repr = typename(node.type)
            return self.node_type_to_id[type_repr]
        else:
            # assert isinstance(node, str)
            # it is a type
            type_repr = typename(node)
            return self.node_type_to_id[type_repr]

    def is_terminal(self, node):
        return node.type in self.terminal_types

    def is_value_node(self, node):
        raise NotImplementedError


PY_AST_NODE_FIELDS = {
    'FunctionDef': {
        'name': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'returns': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'AsyncFunctionDef': {
        'name': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'returns': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'ClassDef': {
        'name': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'bases': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'keywords': {
            'type': ast.keyword,
            'is_list': True,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'Return': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Delete': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Assign': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'AugAssign': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'AnnAssign': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'annotation': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'simple': {
            'type': int,
            'is_list': False,
            'is_optional': False
        }
    },
    'For': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'AsyncFor': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'While': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'If': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'With': {
        'items': {
            'type': ast.withitem,
            'is_list': True,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'AsyncWith': {
        'items': {
            'type': ast.withitem,
            'is_list': True,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'Raise': {
        'exc': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'cause': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Try': {
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'handlers': {
            'type': ast.excepthandler,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'finalbody': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'Assert': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'msg': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Import': {
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        }
    },
    'ImportFrom': {
        'module': {
            'type': str,
            'is_list': False,
            'is_optional': True
        },
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        },
        'level': {
            'type': int,
            'is_list': False,
            'is_optional': True
        }
    },
    'Global': {
        'names': {
            'type': str,
            'is_list': True,
            'is_optional': False
        },
    },
    'Nonlocal': {
        'names': {
            'type': str,
            'is_list': True,
            'is_optional': False
        },
    },
    'Expr': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Pass': { },
    'Break': { },
    'Continue': { },
    'BoolOp': {
        'op': {
            'type': ast.boolop,
            'is_list': False,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'BinOp': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'right': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'UnaryOp': {
        'op': {
            'type': ast.unaryop,
            'is_list': False,
            'is_optional': False
        },
        'operand': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Lambda': {
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'IfExp': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'orelse': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Dict': {
        'keys': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Set': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'ListComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'SetComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'DictComp': {
        'key': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'GeneratorExp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'Await': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Yield': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'YieldFrom': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Compare': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ops': {
            'type': ast.cmpop,
            'is_list': True,
            'is_optional': False
        },
        'comparators': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Call': {
        'func': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'keywords': {
            'type': ast.keyword,
            'is_list': True,
            'is_optional': False
        }
    },
    'Num': {
        'n': {
            'type': object,  #FIXME: should be float or int?
            'is_list': False,
            'is_optional': False
        }
    },
    'Str': {
        's': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'FormattedValue': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'conversion': {
            'type': int,
            'is_list': False,
            'is_optional': True
        },
        'format_spec': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'JoinedStr': {
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'Bytes': {
        's': {
            'type': bytes,
            'is_list': False,
            'is_optional': False
        }
    },
    'NameConstant': {
        'value': {
            'type': object,
            'is_list': False,
            'is_optional': False
        }
    },
    'Ellipsis': { },
    'Constant': {
        'value': {
            'type': object,
            'is_list': False,
            'is_optional': False
        }
    },
    'Attribute': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'attr': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Subscript': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'slice': {
            'type': ast.slice,
            'is_list': False,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Starred': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'slice': {
            'type': ast.slice,
            'is_list': False,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        }
    },
    'Name': {
        'id': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'List': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Tuple': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'ExceptHandler': {
        'type': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'name': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'arguments': {
        'args': {
            'type': ast.arg,
            'is_list': True,
            'is_optional': False
        },
        'vararg': {
            'type': ast.arg,
            'is_list': False,
            'is_optional': True
        },
        'kwonlyargs': {
            'type': ast.arg,
            'is_list': False,
            'is_optional': True
        },
        'kw_defaults': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'kwarg': {
            'type': ast.arg,
            'is_list': False,
            'is_optional': True
        },
        'defaults': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'arg': {
        'arg': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'annotation': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'comprehension': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ifs': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'keyword': {
        'arg': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'alias': {
        'name': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'asname': {
            'type': str,
            'is_list': False,
            'is_optional': True
        }
    },
    'withitem': {
        'context_expr': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'optional_vars': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Slice': {
        'lower': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'upper': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'step': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'ExtSlice': {
        'dims': {
            'type': ast.slice,
            'is_list': True,
            'is_optional': False
        }
    },
    'Index': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'root': {}
}

NODE_FIELD_BLACK_LIST = {'ctx'}

TERMINAL_AST_TYPES = {
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Add,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Div,
    ast.FloorDiv,
    ast.LShift,
    ast.Mod,
    ast.Mult,
    ast.Pow,
    ast.Sub,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.Lt,
    ast.LtE,
    ast.NotEq,
    ast.NotIn,
    ast.Not,
    ast.USub
}


def is_builtin_type(x):
    return x == str or x == int or x == float or x == bool or x == object or x == 'identifier'


def is_terminal_ast_type(x):
    if inspect.isclass(x) and x in TERMINAL_AST_TYPES:
        return True

    return False


def type_str_to_type(type_str):
    if type_str.endswith('*') or type_str == 'root' or type_str == 'epsilon':
        return type_str
    else:
        try:
            type_obj = eval(type_str)
            if is_builtin_type(type_obj):
                return type_obj
        except:
            pass

        try:
            type_obj = eval('ast.' + type_str)
            return type_obj
        except:
            raise RuntimeError('unidentified type string: %s' % type_str)


def is_compositional_leaf(node):
    is_leaf = True

    for field_name, field_value in ast.iter_fields(node):
        if field_name in NODE_FIELD_BLACK_LIST:
            continue

        if field_value is None:
            is_leaf &= True
        elif isinstance(field_value, list) and len(field_value) == 0:
            is_leaf &= True
        else:
            is_leaf &= False
    return is_leaf


class PythonGrammar(Grammar):
    def __init__(self, rules):
        super(PythonGrammar, self).__init__(rules)

    def is_value_node(self, node):
        return is_builtin_type(node.type)

import json


class NLPRequest:
    def __init__(self, context_name, rev_ids, ip=None):
        """
        Construct a ScoreRequest from parameters.

        :Parameters:
            context_name : str
                The name of the content for the query -- usually a wikidb name
            rev_ids : `iterable` ( `int` )
                A set of revision IDs to score
        """
        self.context_name = context_name
        self.rev_ids = set(rev_ids)
        self.ip = ip

    def __str__(self):
        return self.format()

    def format(self, rev_id=None):
        """
        Fomat a request or a sub-part of a request based on a rev_id and/or
        model_name.  This is useful for logging.
        """
        rev_ids = rev_id if rev_id is not None else set(self.rev_ids)
        common = [self.context_name, rev_ids]

        optional = []
        if self.ip:
            optional.append("ip={0}".format(self.ip))

        return "{0}({1})".format(":".join(repr(v) for v in common),
                                 ", ".join(optional))

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__,
            ", ".join(repr(v) for v in [
                self.context_name,
                self.rev_ids,
                self.model_names,
                "ip={0!r}".format(self.ip)
                ]))

    def to_json(self):
        return {
            'context': self.context_name,
            'rev_ids': list(self.rev_ids),
            'ip': self.ip,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            data['context'],
            set(data['rev_ids']),
            ip=data['ip'])

from flask import request

from ... import util
from ... import preprocessors, responses


def configure(config, bp):

    # /v1/nlp/
    @bp.route("/v1/nlp/", methods=["GET"])
    @preprocessors.nocache
    @preprocessors.minifiable
    def nlp_v1():
        try:
           nlp_request = util.build_nlp_request(request)
        except Exception as e:
            return responses.bad_request(str(e))

        return util.jsonify("test")

    return bp

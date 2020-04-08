from flask import request
from flask_swaggerui import render_swaggerui

from . import scores, spec


def configure(config, bp):

    @bp.route("/v1/", methods=["GET"])
    def v1_index():
        if "spec" in request.args:
            return spec.generate_spec(config)
        else:
            return render_swaggerui(swagger_spec_path="/v1/spec/")

    bp = scores.configure(config, bp)
    bp = spec.configure(config, bp)

    return bp

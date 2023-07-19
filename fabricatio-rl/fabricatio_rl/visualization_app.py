from flask import Flask, render_template, jsonify, make_response
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join

from flask_restful import Resource, Api


class LogPaths:
    def __init__(self, logdir):
        self.active_log: str = logdir
        self.logroot: str = str(Path(logdir).resolve().parent)


class RunList(Resource):
    def __init__(self, **kwargs):
        self.logpaths = kwargs['logdirs']

    def get(self):
        logdirs = os.listdir(self.logpaths.logroot)
        self.logpaths.active_log = join(self.logpaths.logroot, logdirs[0])
        return jsonify(logdirs)


class DataMonitor(Resource):
    def __init__(self, **kwargs):
        self.logpaths = kwargs['logdirs']

    def get(self, state_id):
        if len(os.listdir(self.logpaths.active_log)) > state_id:
            return jsonify([True])
        else:
            return jsonify([False])


class RunManager(Resource):
    def __init__(self, **kwargs):
        self.logpaths = kwargs['logdirs']

    def get(self, logdir):
        self.logpaths.active_log = join(self.logpaths.logroot, logdir)
        return ""


class StateData(Resource):
    def __init__(self, **kwargs):
        self.logpaths = kwargs['logdirs']

    def get(self, state_id):
        filename = join(self.logpaths.active_log, str(state_id) + ".json")
        if os.path.isfile(filename):
            with open(filename) as f:
                data = json.load(f)
                prepare_kpi(data)
                return jsonify(data)


class Index(Resource):
    @staticmethod
    def get():
        headers = {'Content-Type': 'text/html'}
        return make_response(
            render_template("landing_slider.html"), 200, headers)


def prepare_kpi(data):
    # job kpis
    df_kpi_j = pd.DataFrame(data['metrics_jobs']).drop(
        ['Jobs Visible', 'Job Index'], axis=1)
    max_y_j = df_kpi_j.to_numpy().max(initial=-np.inf)
    df_kpi_j_t = df_kpi_j.transpose().reset_index(drop=False)
    json_j_records = df_kpi_j_t.to_json(orient="records")
    parsed_j = json.loads(json_j_records)
    # # machine kpis
    df_kpi_m = pd.DataFrame(data['metrics_machines']).drop(
        ['Machine Index'], axis=1)
    max_y_m = df_kpi_m.to_numpy().max(initial=-np.inf)
    df_kpi_m_t = df_kpi_m.transpose().reset_index(drop=False)
    json_m_records = df_kpi_m_t.to_json(orient="records")
    parsed_m = json.loads(json_m_records)
    # # return
    data['metrics_jobs'] = [max_y_j, parsed_j]  #
    data['metrics_machines'] = [max_y_m, parsed_m]


def create_app(logdir: str):
    app = Flask(__name__,
                template_folder='visualization/templates',
                static_folder='visualization/static')
    paths = LogPaths(logdir)
    api = Api(app)
    api.add_resource(Index, '/')
    api.add_resource(StateData, '/state_data/<int:state_id>',
                     resource_class_kwargs={'logdirs': paths})
    api.add_resource(RunManager, '/set_logdir/<string:logdir>',
                     resource_class_kwargs={'logdirs': paths})
    api.add_resource(DataMonitor, '/update_required/<int:state_id>',
                     resource_class_kwargs={'logdirs': paths})
    api.add_resource(RunList, '/get_run_list',
                     resource_class_kwargs={'logdirs': paths})
    return app

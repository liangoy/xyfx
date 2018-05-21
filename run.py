#!/usr/bin/env python3
import connexion
import logging

# from gevent import monkey
# monkey.patch_all()

from connexion import NoContent

logging.basicConfig(level=logging.INFO)
app = connexion.App('main')
app.add_api('api.yaml')
# set the WSGI application callable to allow using uWSGI:
# uwsgi --http :8080 -w app
application = app.app
if __name__ == '__main__':
    # run our standalone gevent server
    app.run(port=6767, server='gevent')
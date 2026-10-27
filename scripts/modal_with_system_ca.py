#!/usr/bin/env python3
"""Use the managed host's trusted CA store for the official Modal CLI.

TLS verification stays enabled. Proxy settings and endpoint policy are unchanged.
Install modal[api-proxy-support]==1.6.2 and truststore==0.10.5 in the client venv.
"""
import runpy

import truststore


if __name__ == '__main__':
    truststore.inject_into_ssl()
    runpy.run_module('modal',run_name='__main__')

# gunicorn_config.py
import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 2
timeout = 120
keepalive = 5

# Important: Enable proxy protocol headers
forwarded_allow_ips = '*'
proxy_protocol = True
proxy_allow_ips = '*'
"""New log formats (owner 2026-10-24, from the attack_data dev half): Suricata http
events, NGINX key="value" access logs, AWS Security Lake (OCSF) CloudTrail."""

from __future__ import annotations

import json

from src.soc.parsers import parse_line


def test_suricata_http_event_is_a_web_request() -> None:
    obj = {"timestamp": "2024-08-02T14:21:23.99+0000", "event_type": "http", "src_ip": "10.3.10.8",
           "dest_ip": "10.3.99.1", "dest_port": 80,
           "http": {"hostname": "10.3.99.1", "url": "/index.html", "http_method": "GET", "status": 200,
                    "http_user_agent": "Mozilla/5.0", "protocol": "HTTP/1.1"}}  # fmt: skip
    ev = parse_line(json.dumps(obj))
    assert ev is not None and ev["source"] == "suricata_http" and ev["kind"] == "http_request"
    assert ev["src_ip"] == "10.3.10.8" and "/index.html" in ev["detail"]


def test_nginx_plus_kv_access_line() -> None:
    line = ('site="shop.example.com" server="3.8.2.2" dest_port="80" src="198.51.100.4" src_ip="198.51.100.4" '
            'user="-" time_local="29/Jan/2024:16:08:00 +0000" protocol="HTTP/1.1" status="200" '
            'http_method="GET" uri_path="/products" uri_query="page=2" http_referer="-" '
            'http_user_agent="Mozilla/5.0"')  # fmt: skip
    ev = parse_line(line)
    assert ev is not None and ev["source"] == "web" and ev["src_ip"] == "198.51.100.4"
    assert "/products?page=2" in ev["detail"] and not ev["techniques"]


def test_ocsf_cloudtrail_becomes_cloudtrail() -> None:
    obj = {"metadata": {"product": {"name": "CloudTrail", "vendor_name": "AWS"}}, "time": 1733921670000,
           "api": {"operation": "ListBuckets", "service": {"name": "s3.amazonaws.com"}, "response": {"error": ""}},
           "actor": {"user": {"type": "IAMUser", "uid": "arn:aws:iam::1:user/alice", "name": "alice"}},
           "src_endpoint": {"ip": "203.0.113.20"}, "cloud": {"region": "us-east-1"}}  # fmt: skip
    ev = parse_line(json.dumps(obj))
    assert ev is not None and ev["source"] == "cloudtrail" and ev["kind"] == "ListBuckets"
    assert ev["src_ip"] == "203.0.113.20"

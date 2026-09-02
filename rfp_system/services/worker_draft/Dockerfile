ARG BASE=rfp-base:local
FROM ${BASE}

COPY services/worker_draft/requirements.txt /tmp/req.txt
RUN python -m pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

COPY services/worker_draft/handler.py ${LAMBDA_TASK_ROOT}/handler.py
CMD ["handler.handler"]

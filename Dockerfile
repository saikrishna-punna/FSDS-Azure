FROM python:3.12-slim

RUN apt-get update -y && \
    apt-get install -y curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
USER appuser

COPY --chown=appuser:appuser ./dist/python_project_uv-0.1.0-py3-none-any.whl /home/appuser/app/python_project_uv-0.1.0-py3-none-any.whl
COPY --chown=appuser:appuser ./main.py /home/appuser/app/main.py

WORKDIR /home/appuser/app/

RUN pip install python_project_uv-0.1.0-py3-none-any.whl

ENTRYPOINT [ "python", "/home/appuser/app/main.py"]
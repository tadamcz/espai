FROM python:3.11.9 AS python

# Check consistency of Python version with .tool-versions
# .tool-versions is only used for local (non-Docker) development,
# so this wouldn't affect the Docker image. The point is to fail loudly
# to let us know they've become out of sync.
COPY .tool-versions .tool-versions
RUN grep -q "^python $(python --version | cut -d' ' -f2)" .tool-versions

FROM python AS dependency-resolver

ENV POETRY_VERSION=1.8.2
ENV POETRY_PLUGIN_EXPORT_VERSION=1.5.0

RUN pip install "poetry==$POETRY_VERSION"
RUN pip install "poetry-plugin-export==$POETRY_PLUGIN_EXPORT_VERSION"

# Dependencies first, to make efficient use of Docker cache
COPY pyproject.toml poetry.lock ./

# Required for poetry check to pass
COPY README.md README.md

# Check lockfile is consistent with pyproject.toml
RUN poetry check

# Use poetry for dependency resolution (but not for managing environments or installing packages)
RUN poetry export --without-hashes --with=test > requirements.txt

FROM python AS dependencies

COPY --from=dependency-resolver /requirements.txt .

# Install dependencies using pip
RUN pip install -r requirements.txt

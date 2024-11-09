FROM ubuntu:latest
LABEL authors="tolik"

ENTRYPOINT ["top", "-b"]
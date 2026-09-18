FROM nvcr.io/nvidia/pytorch:24.07-py3

EXPOSE 28000

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y python3-tk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/diffusion-trainer-studio

# Build the exact checkout supplied as the Docker build context. This keeps
# feature branches, tags and historical commits reproducible instead of silently
# cloning whatever happens to be the repository's default branch at build time.
COPY . .

# The repository relies on pinned submodules for the GUI, tag editor and Anima
# trainers. Docker cannot initialize submodules that were omitted from the build
# context, so fail early with an actionable error instead of producing a broken
# runtime image.
RUN test -f frontend/dist/index.html && \
    test -f mikazuki/dataset-tag-editor/scripts/launch.py && \
    test -f sd-scripts/anima_train_network.py || \
    (echo 'Required submodules are missing. Run: git submodule update --init --recursive' >&2; exit 1)

RUN pip install xformers==0.0.27.post2 --no-deps && \
    pip install -r requirements.txt

CMD ["python", "gui.py", "--listen"]

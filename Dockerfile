FROM mambaorg/micromamba:1.4.9

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER docker_env.yml environment.yml


RUN micromamba install -yn base -f environment.yml && \
    micromamba clean -qafy

RUN mkdir data  log_smhi log_skogs

ENV PATH="/opt/conda/bin:$PATH"
RUN cd data && gdown 19MBh9JIJTxYIPAeO7G5RML5_ddjJ1Cpa && gdown 1fmdHZLD44c2_rmwQh5cskGLb3_Do_Zbl 
RUN cd data && unzip \*.zip && rm *.zip && mv skogsstyrelsen-data skogsstyrelsen && cd -

#COPY --chown=$MAMBA_USER:$MAMBA_USER . ./sdl2-hackathon

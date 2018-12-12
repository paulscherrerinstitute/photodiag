FROM continuumio/miniconda3

EXPOSE 5006

RUN conda install photodiag -c paulscherrerinstitute

CMD source activate base; photodiag palm --args bernina

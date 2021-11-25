Author: Samuel Wallan swallan@calpoly.edu

# Instructions to run program:

Run:
```angular2html
python3 PageRank.py <dataSource> <d> <epsilon> <sink_fix>
```
sink fix: 
- 0: sink -> sink
- 1: sink -> all origins

Full output is saved to `out/pageRank_d<d>_eps<eps>_sparse_<dataSource>.txt"`

Example:
```angular2html
python3 PageRank.py data/NCAA_football.csv .85 0.000000001 0

> 10 sinks, fixing with method <sink => sink>
> Running pagerank....done. time elapsed: 0.00188899
> Running PageRank with data/NCAA_football.csv
> d=0.85, eps=1e-09
> 1537 edges.
> readTime: 0.01s, processTime: 0.00s
> After n=89 iterations:
> 0.10648576 : Utah 0
> 0.02989696 : Mississippi 1
> 0.02385883 : Florida 2
> 0.01523324 : Oklahoma 3
...
```
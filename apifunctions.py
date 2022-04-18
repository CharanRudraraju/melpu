def getMovieInfo(id):
    movie_info = {}
    with open("./movielens/ml-1m/movies_extrainfos.dat", encoding="utf-8") as f:
        for line in f.readlines():
            tmp = line.strip().split("::")
            movie_info[tmp[0]] = "{} ({})".format(tmp[1], tmp[2])
    return movie_info[id]
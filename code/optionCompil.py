from optparse import OptionParser

def OptionCompilation():
    #Lecture des options à l'exécution
    parser = OptionParser()
    parser.add_option("-E", "--epochs", type="int", dest="epochs",
                        help="Nombre d epoques")
    parser.add_option("-m", "--minib", type="int", dest="minibatch",
                        help="Taille du minibatch")
    parser.add_option("-c", "--crop", type="int", dest="cropsize",
                        help="Nombre d epoques")
    parser.add_option("-t", "--lentrain", type="float", dest="lengthtrain",
                        help="Taille du minibatch")
    (options, args) = parser.parse_args()
    if options.epochs and options.minibatch and options.cropsize and options.lengthtrain:
        print("Nombre d epoques = ", options.epochs)
        print("Taille du minibatch = ", options.minibatch)
        print("Taille de resize pour uniformiser les donnees = ", options.cropsize)
        print("Taille du training set = ", options.lengthtrain)
    else:
        parser.error("Pas assez d arguments, demander --help")

    return options

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import math
import copy
import PySide2.QtCore as qtcore
from PySide2.QtWidgets import QWidget, QApplication, QFileDialog
from PySide2.QtPrintSupport import QPrinter
import PySide2.QtWidgets as QtWidgets
from PySide2.QtGui import QPainter, QFont, QColor
import traceback
import argparse
import random

class graphique:
    class Afficheur(QWidget):

        afficheurs = []
        size = (800, 600)
        coord = np.array((50, 50))
        decalage = np.array((40, 40))

        def __init__(
            self,
            sujet,
            centre,
            ech,
            ):
            super(graphique.Afficheur, self).__init__()
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            desktop = app.desktop()
            screen = desktop.screenGeometry(desktop.screenNumber(self))

            self.rayon = 4
            self.ech = ech
            self.centre = np.array(centre)
            self.crayon = None
            self.sujet = sujet
            self.setStyleSheet('background-color: black')
            self.center = np.array(graphique.Afficheur.size) / 2.
            self.setGeometry(graphique.Afficheur.coord[0] + screen.x(),
                            graphique.Afficheur.coord[1] + screen.y(),
                            graphique.Afficheur.size[0], graphique.Afficheur.size[1])
            graphique.Afficheur.coord += graphique.Afficheur.decalage
            self.show()

        def __pointVersPixel(self, p):
            p = np.array(p)
            pixel = (p - self.centre) * self.ech

            # pixel = p * self.ech

            pixel[1] = -pixel[1]

            pixel += self.center
            return qtcore.QPoint(pixel[0], pixel[1])

        def __pixelVersPoint(self, pixel):
            p = np.array((pixel.x(), pixel.y())) - self.center
            p[1] = -p[1]
            p = p / self.ech + self.centre
            return p

        def __pixelVersVecteur(self, pixel1, pixel2):
            v = np.array((pixel2.x(), pixel2.y())) - np.array((pixel1.x(),
                    pixel1.y()))
            v[1] = -v[1]
            v = v / self.ech
            return v

        def tracePoint(self, p):
            pixel = self.__pointVersPixel(p)
            self.crayon.drawEllipse(pixel, self.rayon, self.rayon)

        def traceLigne(self, p1, p2):
            pixel1 = self.__pointVersPixel(p1)
            pixel2 = self.__pointVersPixel(p2)
            self.crayon.drawLine(pixel1, pixel2)

        def changeCouleur(self, couleur):
            c = QColor(couleur[0] * 255, couleur[1] * 255, couleur[2] * 255)
            self.crayon.setPen(c)
            self.crayon.setBrush(c)

        def traceTexte(self, p, texte):
            pixel = self.__pointVersPixel(p)
            pixel += qtcore.QPoint(10, -10)
            self.crayon.drawText(pixel, texte)

        def renomme(self, titre):
            self.setWindowTitle(titre)

        def mousePressEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.clic = pos
            pos = self.__pixelVersPoint(pos)
            self.centre = self.centre

        def mouseMoveEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.centre -= self.__pixelVersVecteur(self.clic, pos)
            self.clic = pos
            self.update()

        def mouseReleaseEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.centre -= self.__pixelVersVecteur(self.clic, pos)
            self.clic = pos
            self.update()

        def wheelEvent(self, event):
            pos = event.pos()
            C = self.__pixelVersPoint(pos)
            k1 = self.ech
            k2 = k1 * math.exp(0.001 * event.delta())
            self.centre = (self.centre * k1 + C * (k2 - k1)) / k2
            self.ech = k2
            self.update()

        def paintEvent(self, event):
            if self.crayon != None:
                return
            self.crayon = QPainter(self)
            self.trace()
            self.crayon = None

        def trace(self):
            self.crayon.setFont(QFont('Decorative', 10))
            if self.sujet != None:
                self.sujet.trace(self)
            L = 100
            x = 20
            y = 20
            self.crayon.setPen('white')
            msg = '{0:.2f} km'.format(L / self.ech)
            l = self.crayon.fontMetrics().boundingRect(msg).width()
            self.crayon.drawText(x + (L - l) / 2, y - 2, msg)
            self.crayon.drawLine(x, y, x + L, y)
            self.crayon.drawLine(x, y - L / 20, x, y + L / 10)
            self.crayon.drawLine(x + L, y - L / 20, x + L, y + L / 10)

        def sauvegarde(self):
            filename = QFileDialog.getSaveFileName(self, 'Fichier PDF')
            if filename:
                printer = QPrinter(QPrinter.HighResolution)
                printer.setPageSize(QPrinter.A4)
                printer.setColorMode(QPrinter.Color)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(filename)
                self.crayon = QPainter(printer)
                self.trace()
                self.crayon = None


    def affiche(
        graphe,
        centre,
        ech,
        blocage=True,
        titre='Quentin Fuxa - Présentation 9/11/18',
        ):
        '''Affiche dans une fenêtre le graphe'''

        graphe = copy.deepcopy(graphe)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        a = graphique.Afficheur(graphe, centre, ech)
        graphique.Afficheur.afficheurs.append(a)
        a.renomme(titre)
        if blocage:
            graphique.bloque()
        return a


    def bloque():
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        app.exec_()

class graphe:
    class Sommet:

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.aretes = []
            self.numero = None
            self.couleur = (0., 0., 0.)
            self.cumul = sys.float_info.max  # par defaut( aant Dijkstra ), le cumul est infini
            self.chemin = None
            self.label = None

        def __str__(self):
            '''Affiche les coordonnees du sommet'''

            return 'v' + str(self.numero) + '(x=' + str(self.x) + 'km  y=' \
                + str(self.y) + 'km )'

        def distance_2(self, v):
            '''calcule la distance au carree'''

            x1 = self.x
            x2 = v.x
            y1 = self.y
            y2 = v.y
            return (x2 - x1) ** 2 + (y2 - y1) ** 2

        def distance(self, v):
            return math.sqrt(self.distance_2(v))


    class Arete:

        def __init__(
            self,
            s1,
            s2,
            longueur,
            vitesse,
            ):

            self.vitesse = vitesse
            self.longueur = longueur
            self.sommet1 = s1
            self.sommet2 = s2
            self.couleur = (0., 0., 0.)
            self.cout = 1

        def voisin(self, s):
            if self.sommet1 == s:
                return self.sommet2
            else:
                return self.sommet1

        def __str__(self):
            '''Affiche des informations sur le sommet'''

            return '{' + str(self.sommet1) + ',' + str(self.sommet2) \
                + '} (long.=' + str(self.longueur) + ' vlim. =' \
                + str(self.vitesse) + 'km/h)'


    class Graphe:

        def __init__(self, nom):
            self.sommets = []
            self.nom = nom
            self.aretes = []

        def ajouteSommet(self, x, y):
            s = graphe.Sommet(x, y)

            # s.numero=numero

            self.sommets.append(s)
            return s

        def connecte(
            self,
            s1,
            s2,
            longueur,
            vitesse,
            ):

            a = graphe.Arete(s1, s2, longueur, vitesse)
            self.aretes.append(a)
            s1.aretes.append(a)
            s2.aretes.append(a)
            return a

        def n(self):
            '''renvoie le nombre de sommets'''

            return len(self.sommets)

        def m(self):
            '''renvoie le nombre d'aretes'''

            m = 0
            for s in self.sommets:
                m += len(s.aretes)
            return m / 2

        def fixeCarburantCommeCout(self):
            for a in self.aretes:
                a.cout = a.longueur

        def fixeTempsCommeCout(self):
            for a in self.aretes:
                a.cout = a.longueur / a.vitesse

        def __str__(self):
            '''Affiche des informations sur tous les sommets et aretes du graphe'''

            textsommets = ''
            for s in self.sommets:
                textsommets += s.__str__() + '\n'
            textaretes = ''
            for v in self.aretes:
                textaretes += v.__str__() + '\n'
            return 'V(Graphe de la figure 1) = { \n' + textsommets \
                + '''} 
    E(Graphe de la figure 1) = { 
    ''' + textaretes \
                + ' \n }'

        def trace(self, afficheur):
            '''Se charge d'afficher le graphe a l'aide de la classe afficheur'''

            for s in self.sommets:
                afficheur.changeCouleur(s.couleur)  # Change les couleurs des  sommets
                if s.numero != None:
                    afficheur.traceTexte((s.x, s.y), 'v' + str(s.numero))
                afficheur.tracePoint((s.x, s.y))
            for a in self.aretes:
                afficheur.changeCouleur(a.couleur)  # Change les couleurs des aretes
                afficheur.traceLigne((a.sommet1.x, a.sommet1.y),
                                    (a.sommet2.x, a.sommet2.y))

        def ajouteRoute(
            self,
            v1,
            v2,
            vmax,
            ):

            longueur = v1.distance(v2)
            return self.connecte(v1, v2, longueur, vmax)

        def ajouteNationale(self, v1, v2):
            '''on colore les nationales en rouge'''

            a = self.ajouteRoute(v1, v2, 90.)
            a.couleur = (1., 0., 0.)
            return a

        def ajouteDepartementale(self, v1, v2):
            '''on colore les nationales en jaune'''

            a = self.ajouteRoute(v1, v2, 60.)
            a.couleur = (1., 1., 0.)
            return a

        def Dijkstra(self, depart):
            '''modifie les attributs cumul et chemin des sommets'''

            for s in self.sommets:
                s.cumul = sys.float_info.max  # on initilise tous les sommets a +l'infinie.
                s.chemin = None
            depart.cumul = 0
            L = [depart]  # L est une liste qui contient tous les sommets voisins
            while len(L) > 0:
                min_cumul = L[0].cumul
                imin = 0
                for (idx, s) in enumerate(L[1:]):  # Dans cette boucle on cherche l'indice du minimum et le minimum de L
                    if s.cumul < min_cumul:
                        min_cumul = s.cumul
                        imin = idx + 1  # pas idx car on est dans L[1:]
                s = L.pop(imin)  # On extrait le minimum  de L ce qui correspond au couut minimum
                for e in s.aretes:  # Pour tous les voisins de ce minimum, on regarde si le cout en passant par s est plus petit. Si oui, on change le chemin et le cumul de ce sommet
                    sp = e.voisin(s)
                    raccourci = s.cumul + e.cout
                    if raccourci < sp.cumul:
                        if sp.cumul == sys.float_info.max:
                            L.append(sp)  # si sp n'a pas encore ete visite, on l'ajoute a L
                        sp.chemin = e
                        sp.cumul = raccourci

        def traceArbreDesChemins(self):
            '''Trace les chemins optimaux en couleur'''

            for s in self.sommets:
                if s.chemin is None:
                    s.couleur = (0., 1., 0.)  # s est le sommet de depart.
                else:
                    s.chemin.couleur = (0., 1., 0.)

        def cheminoptimal(self, arrivee):
            '''renvoie une liste contenant les aretes du chemin optimal entre le d\
            epart et l'arivee'''

            La = [arrivee.chemin]
            s = arrivee
            while s.chemin != None and s.chemin.sommet1.chemin != None \
                and s.chemin.sommet2.chemin != None:  # Tant que l'un des sommets de l'arete n'est pas le sommet de depart

                if s.chemin.sommet1 == s:  # on test dans le tuple (s1,s2) quel est le sommet parent
                    s = s.chemin.sommet2
                else:
                    s = s.chemin.sommet1

                La.append(s.chemin)  # La contient la liste de toutes les aretes pour aller de l'arrivee au depart
            La.reverse()
            return La

        def colorieChemin(self, chemin, c):
            '''colorie le chemin contenue dans la liste chemin'''

            b = 0 
            s1 = chemin[-1].sommet1
            s2 = chemin[-1].sommet2
            s3 = chemin[0].sommet1
            s4 = chemin[0].sommet2
            for a in chemin: 
                a.couleur = c
            for i in s1.aretes:  
                if i in chemin:
                    b = b + 1
            if b == 1:
                s1.couleur = (1., 0., 1.)
            else:
                s2.couleur = (1., 0., 1.)
            b = 0
            for i in s3.aretes:
                if i in chemin:
                    b = b + 1
            if b == 1:
                s3.couleur = (1., 1., 1.)
            else:
                s4.couleur = (1., 1., 1.)

        def matriceCout(self, tournee):
            '''retourne la matrice de coefficient Cij= le cout pour aller de i a j'''

            n = len(tournee)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    self.Dijkstra(tournee[i])  # on recalcule Dijkstra pour chaque nouveau sommet i.
                    L = self.cheminoptimal(tournee[j])  # On ajoute dans L les chemins optimaux pour les j arrivees differentes.
                    s = 0
                    for l in L:
                        if l != None:
                            s += l.cout
                    M[i, j] = s
                    M[j, i] = s  # M est symetrique, donc on recopie le triangle sup dans le triangle inf
            return M

        def voyageurDeCommerceNaif(self, tournee):
            n = len(tournee)
            meilleuritineraire = []
            mincout = sys.float_info.max
            c = 0
            itinerairecourant = [0]
            M = self.matriceCout(tournee)
            t = [False for i in range(n)]

            def backtrack():

                nonlocal t, mincout, meilleuritineraire, M, n, c

                if len(itinerairecourant) == n:
                    k = itinerairecourant[-1]
                    c += M[0, k]
                    if c < mincout:
                        mincout = c
                        meilleuritineraire = [x for x in itinerairecourant]
                    c -= M[0, k]
                else:
                    for i in range(1, n):
                        if not t[i]:
                            j = itinerairecourant[-1]
                            itinerairecourant.append(i)
                            c += M[i, j]
                            t[i] = True
                            backtrack()
                            d = itinerairecourant[-1]
                            itinerairecourant.pop()
                            t[d] = False
                            c -= M[i, j]

            backtrack()
            return (mincout, meilleuritineraire)

        def traceItineraire(self, itineraire):
            '''rajoute pour chaque sommet de tournee son numero de visite'''

            (minCout, meilleuritineraire) = \
                self.voyageurDeCommerceNaif(itineraire)
            n = len(meilleuritineraire)
            for i in range(n):
                itineraire[meilleuritineraire[i]].numero = i
                itineraire[meilleuritineraire[i]].couleur = (1., 1., 1.)
            for i in range(n - 1):
                self.Dijkstra(itineraire[meilleuritineraire[i]])
                l = self.cheminoptimal(itineraire[meilleuritineraire[i
                                    + 1]])
                for e in l:
                    e.couleur = (1., 1., 1.)
            self.Dijkstra(itineraire[meilleuritineraire[n - 1]])
            L = self.cheminoptimal(itineraire[meilleuritineraire[0]])
            for e in L:
                e.couleur = (1., 1., 1.)


def pointsaleatoires(n, L):
    '''Genere un ensemble de n points aléatoires, et de carré de côté L'''

    g2 = graphe.Graphe('Graphe de la figure 1')
    for k in range(n):  # Les points sont générés dans un carré de côté L.
        x = -L / 2 + L * random.random()
        y = -L / 2 + L * random.random()
        g2.ajouteSommet(x, y)
    return copy.deepcopy(g2)


def gabriel(g):
    '''Construit des routes respectant le critere de Gabriel'''

    vmax = 90
    for (idx, s1) in enumerate(g.sommets):
        for s2 in g.sommets[idx + 1:]:
            xc = (s1.x + s2.x) / 2  # On prend le point central entre s1 et s2.
            yc = (s1.y + s2.y) / 2
            ray_2 = s1.distance_2(s2) * .25
            presencepoint = False
            for s3 in g.sommets:
                if s3 not in [s1, s2] and (s3.x - xc) ** 2 + (s3.y
                        - yc) ** 2 < ray_2:  # On vérifie qu'il n'y ai pas de points dans le cercle de rayon s1 s2 et centré sur le milieu de s1 et s2
                    presencepoint = True
                    break
            if not presencepoint:
                g.ajouteRoute(s1, s2, vmax)


def gvr(g):
    '''Construit des routes respectant le critere de GVR'''

    vmax = 90
    for (idx, s1) in enumerate(g.sommets):
        for s2 in g.sommets[idx + 1:]:
            ray_2 = s1.distance_2(s2)  # ray_2 est la distance entre s1 et s2 au carré
            presencepoint = False
            for s3 in g.sommets:
                if s3 not in [s1, s2] and (s3.x - s1.x) ** 2 + (s3.y
                        - s1.y) ** 2 < ray_2 and (s3.x - s2.x) ** 2 \
                    + (s3.y - s2.y) ** 2 < ray_2:  # On vérifie qu'il n'y ait pas de point dans l'intersection des  deux cercles centrés sur s1 et s2, et de rayons [s1 s2].
                    presencepoint = True
                    break
            if not presencepoint:
                g.ajouteRoute(s1, s2, vmax)


def reseau(g):
    '''trace les aretes par  Gabriel et GVR, et trace les routes en fonction\
         des criteres respectés: nationales pour gvr et départementales pour gabriel'''

    for (idx, s1) in enumerate(g.sommets):
        for s2 in g.sommets[idx + 1:]:
            xc = (s1.x + s2.x) / 2
            yc = (s1.y + s2.y) / 2
            ray_2gvr = s1.distance_2(s2)
            ray_2gab = s1.distance_2(s2) * .25
            presencepointgvr = False
            presencepointgab = False
            for s3 in g.sommets:
                if s3 not in [s1, s2] and (s3.x - s1.x) ** 2 + (s3.y
                        - s1.y) ** 2 < ray_2gvr and (s3.x - s2.x) ** 2 \
                    + (s3.y - s2.y) ** 2 < ray_2gvr:
                    presencepointgvr = True  # On vérifie que le point est dans gvr
                if s3 not in [s1, s2] and (s3.x - xc) ** 2 + (s3.y
                        - yc) ** 2 < ray_2gab:
                    presencepointgab = True  # Et s'il est dans gabriel
            if not presencepointgvr:
                g.ajouteDepartementale(s1, s2)  # Si seulement si gabriel, alors c'est une départementale.
            if not presencepointgab and presencepointgvr:
                g.ajouteNationale(s1, s2)  # Sinon c'est une nationale.


def genere_map(n=100, L=20):
    ''' Génère une carte d'adresses situées de manière alétoire.

            Returns
            -------
            le graphe de la carte, avec les nationales comme étant les arêtes\
                 de GVR(g) et les départementales les arêtes de GG(g).
    '''

    g = pointsaleatoires(n, L)
    reseau(g)
    graphique.affiche(g, (3., 2.), 100., blocage=False,
                      titre='Carte générée (rouge: nationale - 90km/h, jaune:\
                           départementale - 60km/h)'
                      )
    return g


def compare_map_cout(g):
    '''Compare les graphes ayant carburant puis temps comme coût.'''

    for a in g.aretes:
        g.fixeCarburantCommeCout()
    g.Dijkstra(g.sommets[0])
    g.traceArbreDesChemins()
    graphique.affiche(g, (3., 2.), 100., blocage=False,
                      titre='Coût : Carburant')
    for a in g.aretes:
        g.fixeTempsCommeCout()
    g.Dijkstra(g.sommets[0])
    g.traceArbreDesChemins()
    graphique.affiche(g, (3., 2.), 100., titre='Coût : Temps')


def calcul_itineraire(g, sommet_depart=0, sommet_arrive=1):
    '''On colorie en vert le chemin optimal'''

    # gvr(g)

    g.fixeCarburantCommeCout()
    g.Dijkstra(g.sommets[sommet_depart])
    g.colorieChemin(g.cheminoptimal(g.sommets[sommet_arrive]), (0., 1.,
                    0.))
    graphique.affiche(g, (3., 2.), 100.,
                      titre='Itinéraire optimal entre le sommet %s et le sommet %s'
                       % (sommet_depart, sommet_arrive))


def differentes_adresses_backtracking(g, nbclients):
    '''test du backtracking'''

    g.fixeCarburantCommeCout()
    tournee = []
    for k in range(nbclients):  # On prend les nbclients premiers sommets
        tournee.append(g.sommets[k])
    g.traceItineraire(tournee)
    graphique.affiche(g, (3., 2.), 100.,
                      titre='Tournée optimale du facteur')


if __name__ == '__main__':
    # graphe = graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', type=int, default=0,
                        help="0: affiche les meilleurs chemins avec les deux\
                             fonctions de coûts (essence, temps). 1:affiche le\
                             chemin optimal entre deux points. 2: affiche le\
                             chemin optimal à prendre pour effectuer une tournée"
                        , choices=range(3))
    parser.add_argument('-n', '--nb_points', type=int, default=100,
                        help="nombre de points aléatoires")
    parser.add_argument('-L', '--largeur', type=int, default=6,
                        help="largeur du carré")
    parser.add_argument('-c', '--nbclients', type=int, default=5,
                        help="Dans le cas de l'option 2 : nombre de clients à \
                            visiter dans la tournée"
                        )
    args = parser.parse_args()
    g = genere_map(args.nb_points, args.largeur)
    if args.option == 0:
        compare_map_cout(g)
    elif args.option == 1:
        calcul_itineraire(g)
    elif args.option == 2:
        differentes_adresses_backtracking(g, args.nbclients)
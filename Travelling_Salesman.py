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

class plot_graph:
    class Display_area(QWidget):

        displays = []
        size = (800, 600)
        coord = np.array((50, 50))
        decalage = np.array((40, 40))

        def __init__(
            self,
            subject,
            center,
            ech,
            ):
            super(plot_graph.Display_area, self).__init__()
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            desktop = app.desktop()
            screen = desktop.screenGeometry(desktop.screenNumber(self))

            self.radius = 4
            self.ech = ech
            self.center = np.array(center)
            self.cradius = None
            self.subject = subject
            self.setStyleSheet('background-color: grey')
            self.center = np.array(plot_graph.Display_area.size) / 2.
            self.setGeometry(plot_graph.Display_area.coord[0] + screen.x(),
                            plot_graph.Display_area.coord[1] + screen.y(),
                            plot_graph.Display_area.size[0], plot_graph.Display_area.size[1])
            plot_graph.Display_area.coord += plot_graph.Display_area.decalage
            self.show()

        def __pointToPixel(self, p):
            p = np.array(p)
            pixel = (p - self.center) * self.ech

            # pixel = p * self.ech

            pixel[1] = -pixel[1]

            pixel += self.center
            return qtcore.QPoint(pixel[0], pixel[1])

        def __pixelToPoint(self, pixel):
            p = np.array((pixel.x(), pixel.y())) - self.center
            p[1] = -p[1]
            p = p / self.ech + self.center
            return p

        def __pixelToVector(self, pixel1, pixel2):
            v = np.array((pixel2.x(), pixel2.y())) - np.array((pixel1.x(),
                    pixel1.y()))
            v[1] = -v[1]
            v = v / self.ech
            return v

        def tracePoint(self, p):
            pixel = self.__pointToPixel(p)
            self.cradius.drawEllipse(pixel, self.radius, self.radius)

        def addLine(self, p1, p2):
            pixel1 = self.__pointToPixel(p1)
            pixel2 = self.__pointToPixel(p2)
            self.cradius.drawLine(pixel1, pixel2)

        def changecolor(self, color):
            c = QColor(color[0] * 255, color[1] * 255, color[2] * 255)
            self.cradius.setPen(c)
            self.cradius.setBrush(c)

        def traceText(self, p, text):
            pixel = self.__pointToPixel(p)
            pixel += qtcore.QPoint(10, -10)
            self.cradius.drawText(pixel, text)

        def rename(self, title):
            self.setWindowTitle(title)

        def mousePressEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.clic = pos
            pos = self.__pixelToPoint(pos)
            self.center = self.center

        def mouseMoveEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.center -= self.__pixelToVector(self.clic, pos)
            self.clic = pos
            self.update()

        def mouseReleaseEvent(self, QMouseEvent):
            pos = QMouseEvent.pos()
            self.center -= self.__pixelToVector(self.clic, pos)
            self.clic = pos
            self.update()

        def wheelEvent(self, event):
            pos = event.pos()
            C = self.__pixelToPoint(pos)
            k1 = self.ech
            k2 = k1 * math.exp(0.001 * event.delta())
            self.center = (self.center * k1 + C * (k2 - k1)) / k2
            self.ech = k2
            self.update()

        def paintEvent(self, event):
            if self.cradius != None:
                return
            self.cradius = QPainter(self)
            self.trace()
            self.cradius = None

        def trace(self):
            self.cradius.setFont(QFont('Decorative', 10))
            if self.subject != None:
                self.subject.trace(self)
            L = 100
            x = 20
            y = 20
            self.cradius.setPen('white')
            msg = '{0:.2f} km'.format(L / self.ech)
            l = self.cradius.fontMetrics().boundingRect(msg).width()
            self.cradius.drawText(x + (L - l) / 2, y - 2, msg)
            self.cradius.drawLine(x, y, x + L, y)
            self.cradius.drawLine(x, y - L / 20, x, y + L / 10)
            self.cradius.drawLine(x + L, y - L / 20, x + L, y + L / 10)

        def save(self):
            filename = QFileDialog.getSaveFileName(self, 'PDF file')
            if filename:
                printer = QPrinter(QPrinter.HighResolution)
                printer.setPageSize(QPrinter.A4)
                printer.setColorMode(QPrinter.Color)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(filename)
                self.cradius = QPainter(printer)
                self.trace()
                self.cradius = None


    def display(
        graph,
        center,
        ech,
        block=True,
        title='',
        ):
        '''display in a window the graph'''

        graph = copy.deepcopy(graph)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        a = plot_graph.Display_area(graph, center, ech)
        plot_graph.Display_area.displays.append(a)
        a.rename(title)
        if block:
            plot_graph.block()
        return a


    def block():
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        app.exec_()

class graph:
    class vertex:

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.edges = []
            self.number = None
            self.color = (0., 0., 0.)
            self.cumul = sys.float_info.max  # default( Dijkstra ), infinite cumul
            self.connection = None
            self.label = None

        def __str__(self):
            '''display coordinates of vertex'''

            return 'v' + str(self.number) + '(x=' + str(self.x) + 'km  y=' \
                + str(self.y) + 'km )'

        def distance_2(self, v):
            '''calcul squared distance'''

            x1 = self.x
            x2 = v.x
            y1 = self.y
            y2 = v.y
            return (x2 - x1) ** 2 + (y2 - y1) ** 2

        def distance(self, v):
            return math.sqrt(self.distance_2(v))


    class Edge:

        def __init__(
            self,
            s1,
            s2,
            length,
            speed,
            ):

            self.speed = speed
            self.length = length
            self.vertex1 = s1
            self.vertex2 = s2
            self.color = (0., 0., 0.)
            self.cost = 1

        def neighbor(self, s):
            if self.vertex1 == s:
                return self.vertex2
            else:
                return self.vertex1

        def __str__(self):
            '''display information about the vertex'''

            return '{' + str(self.vertex1) + ',' + str(self.vertex2) \
                + '} (long.=' + str(self.length) + ' vlim. =' \
                + str(self.speed) + 'km/h)'


    class Graph:

        def __init__(self, name):
            self.l_vertex = []
            self.name = name
            self.edges = []

        def addvertex(self, x, y):
            s = graph.vertex(x, y)

            # s.number=number

            self.l_vertex.append(s)
            return s

        def connect(
            self,
            s1,
            s2,
            length,
            speed,
            ):

            a = graph.Edge(s1, s2, length, speed)
            self.edges.append(a)
            s1.edges.append(a)
            s2.edges.append(a)
            return a

        def n(self):
            '''return the number of vertex'''

            return len(self.l_vertex)

        def m(self):
            '''return the number of edges'''

            m = 0
            for s in self.l_vertex:
                m += len(s.edges)
            return m / 2

        def fixFuelAsCost(self):
            for a in self.edges:
                a.cost = a.length

        def fixTimeAsCost(self):
            for a in self.edges:
                a.cost = a.length / a.speed

        def __str__(self):
            '''display informations about all  vertex and edges of the graph'''

            textl_vertex = ''
            for s in self.l_vertex:
                textl_vertex += s.__str__() + '\n'
            textdges = ''
            for v in self.edges:
                textdges += v.__str__() + '\n'
            return 'V(graph of figure 1) = { \n' + textl_vertex \
                + '''} 
    E(figure 1 graph) = { 
    ''' + textdges \
                + ' \n }'

        def trace(self, display):
            '''Display graph thanks to display class'''

            for s in self.l_vertex:
                display.changecolor(s.color)  # Change the colors  l_vertex
                if s.number != None:
                    display.traceText((s.x, s.y), 'v' + str(s.number))
                display.tracePoint((s.x, s.y))
            for a in self.edges:
                display.changecolor(a.color)  # Change the colors edges
                display.addLine((a.vertex1.x, a.vertex1.y),
                                    (a.vertex2.x, a.vertex2.y))

        def addconnection(
            self,
            v1,
            v2,
            vmax,
            ):

            length = v1.distance(v2)
            return self.connect(v1, v2, length, vmax)

        def addpath1(self, v1, v2):
            '''color path1s in red'''
            a = self.addconnection(v1, v2, 90.)
            a.color = (1., 0., 0.)
            return a

        def addpath2(self, v1, v2):
            '''color path2s in yellow'''

            a = self.addconnection(v1, v2, 60.)
            a.color = (1., 1., 0.)
            return a

        def Dijkstra(self, depart):
            '''modify labels l_vertex'''

            for s in self.l_vertex:
                s.cumul = sys.float_info.max  # initializes l_vertex with +inf.
                s.connection = None
            depart.cumul = 0
            L = [depart]  # L is a list that contains all l_vertex neighbors
            while len(L) > 0:
                min_cumul = L[0].cumul
                imin = 0
                for (idx, s) in enumerate(L[1:]):  # in this loop we search the index of the minimum of L
                    if s.cumul < min_cumul:
                        min_cumul = s.cumul
                        imin = idx + 1  # no idx because we search in L[1:]
                s = L.pop(imin)  # We extract the minimum  of L which corresponds to minimal cost
                for e in s.edges:  # For all neighbors of this minimum, we look if the cost by using s is lower. if yes, we change the connection.
                    sp = e.neighbor(s)
                    raccourci = s.cumul + e.cost
                    if raccourci < sp.cumul:
                        if sp.cumul == sys.float_info.max:
                            L.append(sp)  # if sp has not been visited yet, we add it to L
                        sp.connection = e
                        sp.cumul = raccourci

        def plotConnectionsTree(self):
            '''Plot optimal connections in yellow'''

            for s in self.l_vertex:
                if s.connection is None:
                    s.color = (0., 1., 0.)  # s is the starting vertex.
                else:
                    s.connection.color = (0., 1., 0.)

        def optimalConnection(self, arrivee):
            '''return a list containing the edges of the optimal connection between start and end apex.'''

            La = [arrivee.connection]
            s = arrivee
            while s.connection != None and s.connection.vertex1.connection != None \
                and s.connection.vertex2.connection != None: 

                if s.connection.vertex1 == s:  # test in the tuple (s1,s2) which is the parent vertex
                    s = s.connection.vertex2
                else:
                    s = s.connection.vertex1

                La.append(s.connection)  # contains the list of all the edges to go from arrival to departure
            La.reverse()
            return La

        def colorConnection(self, connection, c):
            '''color connection(s) in connection list'''

            b = 0 
            s1 = connection[-1].vertex1
            s2 = connection[-1].vertex2
            s3 = connection[0].vertex1
            s4 = connection[0].vertex2
            for a in connection: 
                a.color = c
            for i in s1.edges:  
                if i in connection:
                    b = b + 1
            if b == 1:
                s1.color = (1., 0., 1.)
            else:
                s2.color = (1., 0., 1.)
            b = 0
            for i in s3.edges:
                if i in connection:
                    b = b + 1
            if b == 1:
                s3.color = (1., 1., 1.)
            else:
                s4.color = (1., 1., 1.)

        def matrixcost(self, tour):
            '''return matrix of coefficient Cij = cost of the arc i to j'''

            n = len(tour)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    self.Dijkstra(tour[i])  # we calcul Dijkstra for each new vertex i.
                    L = self.optimalConnection(tour[j])  # we add in L the optimal connections for the j new points.
                    s = 0
                    for l in L:
                        if l != None:
                            s += l.cost
                    M[i, j] = s
                    M[j, i] = s  # M is symetric, so we copy upper triangle in lower triangle
            return M

        def NaiveTravellingSalesman(self, tour):
            n = len(tour)
            bestItinerary = []
            mincost = sys.float_info.max
            c = 0
            currentRoute = [0]
            M = self.matrixcost(tour)
            t = [False for i in range(n)]

            def backtrack():

                nonlocal t, mincost, bestItinerary, M, n, c

                if len(currentRoute) == n:
                    k = currentRoute[-1]
                    c += M[0, k]
                    if c < mincost:
                        mincost = c
                        bestItinerary = [x for x in currentRoute]
                    c -= M[0, k]
                else:
                    for i in range(1, n):
                        if not t[i]:
                            j = currentRoute[-1]
                            currentRoute.append(i)
                            c += M[i, j]
                            t[i] = True
                            backtrack()
                            d = currentRoute[-1]
                            currentRoute.pop()
                            t[d] = False
                            c -= M[i, j]

            backtrack()
            return (mincost, bestItinerary)

        def plotItinerary(self, itinerary):
            '''add visite number on each new vertex'''

            (mincost, bestItinerary) = \
                self.NaiveTravellingSalesman(itinerary)
            n = len(bestItinerary)
            for i in range(n):
                itinerary[bestItinerary[i]].number = i
                itinerary[bestItinerary[i]].color = (1., 1., 1.)
            for i in range(n - 1):
                self.Dijkstra(itinerary[bestItinerary[i]])
                l = self.optimalConnection(itinerary[bestItinerary[i
                                    + 1]])
                for e in l:
                    e.color = (1., 1., 1.)
            self.Dijkstra(itinerary[bestItinerary[n - 1]])
            L = self.optimalConnection(itinerary[bestItinerary[0]])
            for e in L:
                e.color = (1., 1., 1.)


def randomPoints(n, L):
    '''generate a  n points random ensemble, and the L size square'''

    g2 = graph.Graph('figure 1 graph')
    for k in range(n):  # points are generated in a square of side L.
        x = -L / 2 + L * random.random()
        y = -L / 2 + L * random.random()
        g2.addvertex(x, y)
    return copy.deepcopy(g2)


def gabriel(g):
    '''Build connections which respect Gabriel criteria'''

    vmax = 90
    for (idx, s1) in enumerate(g.l_vertex):
        for s2 in g.l_vertex[idx + 1:]:
            xc = (s1.x + s2.x) / 2  # We take central point between s1 and s2.
            yc = (s1.y + s2.y) / 2
            ray_2 = s1.distance_2(s2) * .25
            presencepoint = False
            for s3 in g.l_vertex:
                if s3 not in [s1, s2] and (s3.x - xc) ** 2 + (s3.y
                        - yc) ** 2 < ray_2:  # We verify there is no points in the cercle of radius s1 s2 and centered in the middle between s1 and s2
                    presencepoint = True
                    break
            if not presencepoint:
                g.addconnection(s1, s2, vmax)


def gvr(g):
    '''Build connections which respect GVR criteria'''

    vmax = 90
    for (idx, s1) in enumerate(g.l_vertex):
        for s2 in g.l_vertex[idx + 1:]:
            ray_2 = s1.distance_2(s2)  # ray_2 is the squared distance between s1 and s2
            presencepoint = False
            for s3 in g.l_vertex:
                if s3 not in [s1, s2] and (s3.x - s1.x) ** 2 + (s3.y
                        - s1.y) ** 2 < ray_2 and (s3.x - s2.x) ** 2 \
                    + (s3.y - s2.y) ** 2 < ray_2:  # We verify that there is no point in the intersection of the two circles centered on s1 and s2, and of radius [s1 s2].
                    presencepoint = True
                    break
            if not presencepoint:
                g.addconnection(s1, s2, vmax)


def network(g):
    '''plot the Gabriel and GVR edges, and plot connections given the\
         criteria:  path1s for gvr and path2 for gabriel'''

    for (idx, s1) in enumerate(g.l_vertex):
        for s2 in g.l_vertex[idx + 1:]:
            xc = (s1.x + s2.x) / 2
            yc = (s1.y + s2.y) / 2
            ray_2gvr = s1.distance_2(s2)
            ray_2gab = s1.distance_2(s2) * .25
            presencepointgvr = False
            presencepointgab = False
            for s3 in g.l_vertex:
                if s3 not in [s1, s2] and (s3.x - s1.x) ** 2 + (s3.y
                        - s1.y) ** 2 < ray_2gvr and (s3.x - s2.x) ** 2 \
                    + (s3.y - s2.y) ** 2 < ray_2gvr:
                    presencepointgvr = True  # We verify if the point is in gvr
                if s3 not in [s1, s2] and (s3.x - xc) ** 2 + (s3.y
                        - yc) ** 2 < ray_2gab:
                    presencepointgab = True  # and if it is in gabriel
            if not presencepointgvr:
                g.addpath2(s1, s2)  # If and onlt if gabriel, it is a path_2.
            if not presencepointgab and presencepointgvr:
                g.addpath1(s1, s2)  # otherwise it is a path1.


def generate_map(n=100, L=20):
    ''' Create a adress map with random positions.

            Returns
            -------
            the map graph, with path1s being the summit
                 of GVR(g) and path_2 the edges of GG(g).
    '''

    g = randomPoints(n, L)
    network(g)
    plot_graph.display(g, (3., 2.), 100., block=False,
                      title='created map (red: path1- 90km/h, yellow:\
                           path2 - 60km/h)'
                      )
    print(g.l_vertex)
    return g


def compare_map_cost(g):
    '''Compare graphs with fuel then time as cost.'''

    for a in g.edges:
        g.fixFuelAsCost()
    g.Dijkstra(g.l_vertex[0])
    g.plotConnectionsTree()
    plot_graph.display(g, (3., 2.), 100., block=False,
                      title='cost : Carburant')
    for a in g.edges:
        g.fixTimeAsCost()
    g.Dijkstra(g.l_vertex[0])
    g.plotConnectionsTree()
    plot_graph.display(g, (3., 2.), 100., title='cost : time')


def calcul_itinerary(g, vertex_depart=0, vertex_arrive=1):
    '''Color in green optimal connections'''

    # gvr(g)

    g.fixFuelAsCost()
    g.Dijkstra(g.l_vertex[vertex_depart])
    g.colorConnection(g.optimalConnection(g.l_vertex[vertex_arrive]), (0., 1.,
                    0.))
    plot_graph.display(g, (3., 2.), 100.,
                      title='optimal route between vertex %s and %s'
                       % (vertex_depart, vertex_arrive))


def different_backtracking_addresses(g, nbclients):
    '''backtracking test'''

    g.fixFuelAsCost()
    tour = []
    for k in range(nbclients):  # We take the first nbclients of l_vertex
        tour.append(g.l_vertex[k])
    g.plotItinerary(tour)
    plot_graph.display(g, (3., 2.), 100.,
                      title='Optimal tour')


if __name__ == '__main__':
    # graph = graph()
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-o', '--option', type=int, default=0,
                        help="0: display the best connections with the two\
                             cost fonctions for (fuel, time).\n 1:display\
                             optimal connection  betwen 2 points.\n 2: display\
                             optimal connection to do the tour"
                        , choices=range(3))
    parser.add_argument('-n', '--nb_points', type=int, default=100,
                        help="namebre of random points")
    parser.add_argument('-L', '--width', type=int, default=6,
                        help="width of the square")
    parser.add_argument('-c', '--nbclients', type=int, default=5,
                        help="In the option 2 case: number of clients to \
                            visit in the tour"
                        )
    args = parser.parse_args()
    g = generate_map(args.nb_points, args.width)
    if args.option == 0:
        compare_map_cost(g)
    elif args.option == 1:
        calcul_itinerary(g)
    elif args.option == 2:
        different_backtracking_addresses(g, args.nbclients)
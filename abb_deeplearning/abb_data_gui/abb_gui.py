
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QApplication, \
QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QSlider, QLCDNumber
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('Ready')

        centralWidget = QWidget()

        exitAction = QAction(QIcon('exit.png'), '&Exit', self)

        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        okButton = QPushButton('OK')
        cancelButton = QPushButton("Cancel")
        sld = QSlider(Qt.Horizontal,centralWidget)
        lcd = QLCDNumber(centralWidget)

        hbox = QHBoxLayout()
        
        hbox.addWidget(sld)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)


        vbox = QVBoxLayout()
        vbox.addWidget(lcd)
        vbox.addStretch(1)

        vbox.addLayout(hbox)

        centralWidget.setLayout(vbox)


        sld.valueChanged.connect(lcd.display)


        self.setCentralWidget(centralWidget)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle(
            'ABB Deep Reinforcement Learning: Power grid stabilization')

        self.show()

    def keyPressEvent(self,e):
    	if e.key() == Qt.Key_Escape:
    		self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

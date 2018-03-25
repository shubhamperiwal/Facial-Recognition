from kivy.app import App #renders basic kivy app
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

#Custom import
from Detector import authorise

"""
Defines unlock screen
"""
class UnlockScreen(Screen):
    #Decides wether to show data or not
    def switch(self):
        isAuthorised = authorise()
        if isAuthorised:
            self.parent.current = 'data'
        else:
            self.parent.current = 'unauthorised'

"""
Defines screen shown on successful login
"""
class DataScreen(Screen):
    pass

class Unauthorised(Screen):
    pass

"""
Manages Screens
"""
class AppScreenManager(ScreenManager):
    pass

#Written in Kivy language
root_widget = Builder.load_string('''
#:import FadeTransition kivy.uix.screenmanager.FadeTransition
AppScreenManager:
    transition: FadeTransition()
    UnlockScreen:
    DataScreen:
    Unauthorised:
<UnlockScreen>:
    name: 'unlock'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Get confidential data'
            font_size: 50
        BoxLayout:
            orientation: 'horizontal'
            BoxLayout:
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: 'Unlock'
                    font_size: 30
                    on_release: root.switch()
                BoxLayout:
            BoxLayout:
<DataScreen>:
    name: 'data'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Credit Card Number: 4312765980'
            font_size: 50
        BoxLayout:
            orientation: 'horizontal'
            BoxLayout:
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: 'Lock again'
                    font_size: 30
                    on_release: app.root.current = 'unlock'
                BoxLayout:
            BoxLayout:
<Unauthorised>:
    name: 'unauthorised'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Unauthorised face'
            font_size: 50
        BoxLayout:
            orientation: 'horizontal'
            BoxLayout:
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: 'Go back'
                    font_size: 30
                    on_release: app.root.current = 'unlock'
                BoxLayout:
            BoxLayout:
''')

"""
Renders all screens
"""
class SecurityApp(App):
    #Defines main build i.e. what to display
    def build(self):
        return root_widget

#Run the app
SecurityApp().run()
# Findash
Financial Dashboard for Algo Trading . 

Created : 12th April 2024
Current Collaborators: JaaliDollar , TandaiAnxle

Hey Coders . 
Algorithmic Trading is field cluttered with numerous softwares. Softwares filled with features no one ever uses in their
entire lifetime. I propose Findash as a One-Stop-Solution for Data Scientists interested in Financial Markets.

Retrieval , Cleaning , Visualization , Signal Generation , P&L maintenance , Risk/Reward Management and other 
works all of which require attention could be easily managed through a software filled with only those features
that our Closed Community Members agree upon and contribute. 

A Clean Workstation with various Platforms specialising in various aspects of our Project. Various Platforms can be moved 
around the screen in 3 modes ->  Corner:1,2,3,4 , Half: v_up,v_down,h_right,h_left , FullScreen . 

Here's a rough outline of the various platforms that we are going to build in our app each specialising in their respective 
field of concern. This is aimed to improve both User and Developer Experience. 

1. DATAMAN
   Data Manager Platform/Tab equipped with various tools and functions to help you Play with your data.
   Features:
   1.API integration for Historical OHLC data.
   2.Easy download of data for a Time Period.
   3.Cleaning of data for Duplicates ,empty values
   4.Data Visualization using Plotly .This step should open a new Tab in Graph Platform.
   5.Application of Technical Indicators and Statistical Methods as per choice.
   6.Adjusting parameters of Technical Indicators and Statistical Methods to see their
     effect in Real Time.
      etc...

    Note that DATAMAN isn't meant to be used while Live Trading.
    It's a Platform to play with our data however we like to understand patterns and 
    devise plans. 

2. STRATS
   Strategies developed in DATAMAN can be pushed here, you might decide to make a new strategy of
   your own , you might wanna modify some existing strategies python files.
   This Platforms stores your saved strategies. It will be used to activate/deactivate Indicators and
   other Functions as we like.

3. Text Editor tab ( GVIM/Featherpad )
   This tab enables u to use the text editor of your choice to modify any file in the entire software .
   Just open this Platform/Tab wherever you like ,in the top-right corner or the half of the right side
   of your screen.

4. GRAPH Platform/Tab
   This platform enables us to see our data using python's Plotly Library. It would be used to see
   Candle-Stick Charts , OHLC bars , indicators , certain indicators in a new tab in Graph Platform itself etc.

   Also, Every Platform in the entire software has the ability to use the Graph Platform for visulising any kind of data
   using a new tab in Graph Platform.

5. LIVE Platform .
   This is the Heart of our Software , it's the main thing.
   Instead of opening as a tab on the screen that we can place around the screen as we like, LIVE Platform
   occupies a completly new Work-Screen. In the new Screen we can open old Tabs/Platforms but LIVE Platform comes with
   certain new tabs tailored for Intra-Day Trading preferably in Futures and Options.    

   a few of it's sub-plaforms:

   1. API/WebSocket integration ( for real-time )
      Apart from connecting to the web-socket , this tab also searches for existing historical data ,
      connects the historical data with the real-time data for a longer scrollable CandleStick Chart.
      It's not absolutely necessary but the User should be able to see the Candle Sticks of the past .

      It should save real-time data as they arrive. This real-time data is generally of high frequency
      but can be trimmed to show Candles of different durations like 5-min , 15-min , 5 Days etc.

    2. Live P&L Monitoring
       This is configured differently for every different Strategies.

    3. Risk Management System
    4. Tree View for all the Divisions of Capial with their respective P&L, Risk/Reward statement.


 6. < Contribute more Ideas . It's worth your Time >

Currently we have rough UI 
https://github.com/fincode9798/Findash/blob/main/TandaiWindow

and a backend code for data_cleaning, calculating_indicators, analysing_indicators , generating_signals and data_visualization.
https://github.com/fincode9798/Findash/blob/master/x.py

The project is in it's infant stage rn. 
I welcome every python coder interested in Financial Markets to join and share their ideas only. We'll discuss and 
try come up with better options. Remember this project would create a software handcrafted by this community meant only for
the members of this community. 
Happy Coding !



   Data Manager is the tab/Platform that provides various

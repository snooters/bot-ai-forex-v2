//+------------------------------------------------------------------+
//|                                              BotIndicators.mq5   |
//|                        AI Forex Bot v2 — Indicator Pack          |
//|  Menampilkan semua indikator yang dipakai bot di chart MT5       |
//+------------------------------------------------------------------+
#property copyright "AI Forex Bot v2"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 12
#property indicator_plots   8

//--- Plot setup: EMA20, EMA50, EMA200, VWAP, PivotHigh, PivotLow, Session markers
#property indicator_label1  "EMA20"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

#property indicator_label2  "EMA50"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrOrange
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

#property indicator_label3  "EMA200"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrRed
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

#property indicator_label4  "VWAP"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrLime
#property indicator_style4  STYLE_DOT
#property indicator_width4  1

#property indicator_label5  "PivotHigh"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrRed
#property indicator_width5  2

#property indicator_label6  "PivotLow"
#property indicator_type6   DRAW_ARROW
#property indicator_color6  clrLime
#property indicator_width6  2

#property indicator_label7  "Support"
#property indicator_type7   DRAW_ARROW
#property indicator_color7  clrGray
#property indicator_width7  1

#property indicator_label8  "Resistance"
#property indicator_type8   DRAW_ARROW
#property indicator_color8  clrGray
#property indicator_width8  1

//--- Sub-window buffers untuk RSI, MACD, ADX (not visible in main chart window)
//--- These use separate subwindows, but we'll just mark them as comments for now

//--- Input parameters
input int      EMAPeriod1     = 20;    // EMA 1 Period
input int      EMAPeriod2     = 50;    // EMA 2 Period
input int      EMAPeriod3     = 200;   // EMA 3 Period
input int      PivotLookback  = 10;    // Pivot Lookback (bars)
input bool     ShowVWAP       = true;  // Show VWAP
input bool     ShowSessions   = true;  // Show Session Markers
input bool     ShowPivots     = true;  // Show Market Structure Pivots
input bool     ShowSRLevels   = true;  // Show Support/Resistance
input int      SRPeriod       = 30;    // Support/Resistance lookback

//--- Indicator buffers
double         EMA20Buffer[];
double         EMA50Buffer[];
double         EMA200Buffer[];
double         VWAPBuffer[];
double         PivotHighBuffer[];
double         PivotLowBuffer[];
double         SupportBuffer[];
double         ResistanceBuffer[];

//--- Handles for indicators
int            ema20Handle;
int            ema50Handle;
int            ema200Handle;

//--- Session hours
#define ASIA_START   0   // 00:00 GMT
#define ASIA_END     8   // 08:00 GMT
#define LONDON_START 7   // 07:00 GMT
#define LONDON_END   16  // 16:00 GMT
#define NY_START     13  // 13:00 GMT
#define NY_END       22  // 22:00 GMT

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set indicator buffers
   SetIndexBuffer(0, EMA20Buffer,     INDICATOR_DATA);
   SetIndexBuffer(1, EMA50Buffer,     INDICATOR_DATA);
   SetIndexBuffer(2, EMA200Buffer,    INDICATOR_DATA);
   SetIndexBuffer(3, VWAPBuffer,      INDICATOR_DATA);
   SetIndexBuffer(4, PivotHighBuffer, INDICATOR_DATA);
   SetIndexBuffer(5, PivotLowBuffer,  INDICATOR_DATA);
   SetIndexBuffer(6, SupportBuffer,   INDICATOR_DATA);
   SetIndexBuffer(7, ResistanceBuffer,INDICATOR_DATA);
   
   // Plot 5 = PivotHigh (arrow down in main window)
   PlotIndexSetInteger(4, PLOT_ARROW, 234);  // down arrow
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, 0);
   
   // Plot 6 = PivotLow (arrow up in main window)
   PlotIndexSetInteger(5, PLOT_ARROW, 233);  // up arrow
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, 0);
   
   // Plot 7 = Support levels
   PlotIndexSetInteger(6, PLOT_ARROW, 159);  // bullet
   PlotIndexSetDouble(6, PLOT_EMPTY_VALUE, 0);
   
   // Plot 8 = Resistance levels
   PlotIndexSetInteger(7, PLOT_ARROW, 159);  // bullet
   PlotIndexSetDouble(7, PLOT_EMPTY_VALUE, 0);
   
   //--- Get EMA handles
   ema20Handle = iMA(_Symbol, _Period, EMAPeriod1, 0, MODE_EMA, PRICE_CLOSE);
   ema50Handle = iMA(_Symbol, _Period, EMAPeriod2, 0, MODE_EMA, PRICE_CLOSE);
   ema200Handle = iMA(_Symbol, _Period, EMAPeriod3, 0, MODE_EMA, PRICE_CLOSE);
   
   if(ema20Handle == INVALID_HANDLE || ema50Handle == INVALID_HANDLE || ema200Handle == INVALID_HANDLE)
   {
      Print("Failed to create EMA handles");
      return INIT_FAILED;
   }
   
   IndicatorSetString(INDICATOR_SHORTNAME, "BotIndicators(" + IntegerToString(EMAPeriod1) + "," +
                      IntegerToString(EMAPeriod2) + "," + IntegerToString(EMAPeriod3) + ")");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < EMAPeriod3 + PivotLookback + 10)
      return 0;
   
   int limit = prev_calculated > 0 ? prev_calculated - 1 : PivotLookback + 10;
   if(limit < 0) limit = 0;
   
   //--- Copy EMA data
   if(CopyBuffer(ema20Handle, 0, 0, rates_total, EMA20Buffer) <= 0) return 0;
   if(CopyBuffer(ema50Handle, 0, 0, rates_total, EMA50Buffer) <= 0) return 0;
   if(CopyBuffer(ema200Handle, 0, 0, rates_total, EMA200Buffer) <= 0) return 0;
   
   //--- Calculate VWAP
   if(ShowVWAP)
   {
      double cumPriceVol = 0;
      double cumVol = 0;
      int vwapStart = 0;
      MqlDateTime dt1, dt2;
      
      // VWAP resets daily — find today's first bar
      for(int i = rates_total - 1; i >= 0; i--)
      {
         if(i > 0)
         {
            TimeToStruct(time[i], dt1);
            TimeToStruct(time[i-1], dt2);
            if(dt1.day_of_year != dt2.day_of_year || dt1.year != dt2.year)
            {
               vwapStart = i;
               break;
            }
         }
      }
      
      for(int i = rates_total - 1; i >= vwapStart; i--)
      {
         double typicalPrice = (high[i] + low[i] + close[i]) / 3.0;
         cumPriceVol += typicalPrice * (double)tick_volume[i];
         cumVol += (double)tick_volume[i];
         
         if(cumVol > 0)
            VWAPBuffer[i] = cumPriceVol / cumVol;
         else
            VWAPBuffer[i] = close[i];
      }
   }
   
   //--- Calculate pivot highs/lows (market structure)
   if(ShowPivots)
   {
      for(int i = limit; i < rates_total; i++)
      {
         if(i < PivotLookback || i >= rates_total - PivotLookback)
         {
            PivotHighBuffer[i] = 0;
            PivotLowBuffer[i] = 0;
            continue;
         }
         
         // Pivot High: highest among surrounding bars
         bool isHigh = true;
         for(int j = 1; j <= PivotLookback; j++)
         {
            if(high[i] <= high[i-j] || high[i] <= high[i+j])
            {
               isHigh = false;
               break;
            }
         }
         
         // Pivot Low: lowest among surrounding bars
         bool isLow = true;
         for(int j = 1; j <= PivotLookback; j++)
         {
            if(low[i] >= low[i-j] || low[i] >= low[i+j])
            {
               isLow = false;
               break;
            }
         }
         
         PivotHighBuffer[i] = isHigh ? high[i] : 0;
         PivotLowBuffer[i] = isLow ? low[i] : 0;
      }
   }
   
   //--- Calculate Support / Resistance (simple: recent range extremes)
   if(ShowSRLevels)
   {
      for(int i = limit; i < rates_total; i++)
      {
         if(i < SRPeriod)
         {
            SupportBuffer[i] = 0;
            ResistanceBuffer[i] = 0;
            continue;
         }
         
         double highMax = high[ArrayMaximum(high, i - SRPeriod, SRPeriod)];
         double lowMin = low[ArrayMinimum(low, i - SRPeriod, SRPeriod)];
         
         // Mark support/resistance zones
         SupportBuffer[i] = lowMin;
         ResistanceBuffer[i] = highMax;
      }
   }
   
   //--- Draw session markers as vertical lines
   if(ShowSessions)
   {
      static datetime lastDrawn = 0;
      
      for(int i = 0; i < rates_total; i++)
      {
         MqlDateTime dt;
         TimeToStruct(time[i], dt);
         
         // Only draw on confirmed bars (not the current one)
         if(i >= rates_total - 2)
            continue;
         
         // Don't redraw if already drawn on this bar
         if(time[i] == lastDrawn)
            continue;
         
         int hour = dt.hour;
         
         // Asia session start
         if(hour == ASIA_START && dt.min == 0)
         {
            string name = "Asia_" + IntegerToString(time[i]);
            ObjectCreate(0, name, OBJ_VLINE, 0, time[i], 0);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrYellow);
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetString(0, name, OBJPROP_TEXT, "Asia");
            lastDrawn = time[i];
         }
         
         // London session start
         if(hour == LONDON_START && dt.min == 0)
         {
            string name = "London_" + IntegerToString(time[i]);
            ObjectCreate(0, name, OBJ_VLINE, 0, time[i], 0);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrBlue);
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetString(0, name, OBJPROP_TEXT, "London");
            lastDrawn = time[i];
         }
         
         // NY session start
         if(hour == NY_START && dt.min == 0)
         {
            string name = "NY_" + IntegerToString(time[i]);
            ObjectCreate(0, name, OBJ_VLINE, 0, time[i], 0);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetString(0, name, OBJPROP_TEXT, "NY");
            lastDrawn = time[i];
         }
         
         // Overlap markers (London+NY overlap)
         if(hour == LONDON_START + 6 && dt.min == 0)
         {
            string name = "Overlap_LDN_NY_" + IntegerToString(time[i]);
            ObjectCreate(0, name, OBJ_VLINE, 0, time[i], 0);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrMagenta);
            ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
            ObjectSetString(0, name, OBJPROP_TEXT, "LDN+NY");
            lastDrawn = time[i];
         }
      }
   }
   
   return rates_total;
}
//+------------------------------------------------------------------+

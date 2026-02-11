from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

ALL_TICKERS = """
A       AMAT    AZO     CAG     CMI     CZR     DXCM    EW      GBTC    HBAN    IEX     JNUG    LLY     MHK     NEE     ORLY    POOL    RVTY    STT     TPR     VGT     WRB
AAPL    AMCR    BA      CAH     CMS     D       EA      EXC     GD      HCA     IFF     JPM     LMT     MKC     NEM     OTIS    PPG     SBAC    STX     TQQQ    VICI    WST
ABBV    AMD     BAC     CARR    CNC     DAL     EBAY    EXPD    GDDY    HD      IJR     K       LNT     MKTX    NFLX    OXY     PPL     SBUX    STZ     TRGP    VLO     WTW
ABNB    AME     BALL    CAT     CNP     DAY     ECL     EXPE    GDX     HES     INCY    KDP     LOW     MLM     NI      PANW    PRU     SCHW    SW      TRMB    VLTO    WY
ABT     AMGN    BAX     CB      COF     DD      ED      EXR     GDXJ    HIG     INTC    KEY     LQD     MMC     NKE     PARA    PSA     SHW     SWK     TROW    VMC     WYNN
ACGL    AMP     BBY     CBOE    COO     DE      EFX     F       GE      HII     INTU    KEYS    LRCX    MMM     NLR     PAYC    PSX     SHY     SWKS    TRV     VOO     XEL
ACN     AMT     BDX     CBRE    COP     DECK    EG      FANG    GEHC    HLT     INVH    KHC     LULU    MNST    NOC     PAYX    PTC     SILJ    SYF     TSCO    VRSK    XLK
ADBE    AMZN    BEN     CCI     COR     DELL    EIX     FAST    GEN     HOLX    IP      KIM     LUV     MO      NOW     PCAR    PWR     SJM     SYK     TSLA    VRSN    XLP
ADI     ANET    BF.B    CCL     COST    DFS     EL      FCX     GEV     HON     IPG     KKR     LVS     MOH     NRG     PCG     PYPL    SLB     SYY     TSN     VRTX    XLU
ADM     ANSS    BG      CDNS    CPAY    DG      ELV     FDS     GILD    HOOD    IQV     KLAC    LW      MOS     NSC     PEG     QCOM    SLV     T       TT      VST     XLV
ADP     AON     BIIB    CDW     CPB     DGX     EMB     FDX     GIS     HPE     IR      KMB     LYB     MPC     NTAP    PEP     QLD     SMCI    TAP     TTWO    VTR     XLY
ADSK    AOS     BK      CE      CPRT    DHI     EMN     FE      GL      HPQ     IRM     KMI     LYV     MPWR    NTRS    PFE     QQQ     SNA     TDG     TXN     VTRS    XOM
AEE     APA     BKNG    CEG     CPT     DHR     EMR     FFIV    GLD     HRL     ISRG    KMX     MA      MRK     NUE     PFG     RCL     SNPS    TDY     TXT     VZ      XYL
AEP     APD     BKR     CF      CRL     DIS     ENPH    FI      GLW     HSIC    IT      KO      MAA     MRNA    NUGT    PG      REG     SO      TECH    TYL     WAB     YUM
AES     APH     BLDR    CFG     CRM     DLR     EOG     FICO    GM      HST     ITW     KR      MAR     MS      NVDA    PGR     REGN    SOLV    TEL     UAL     WAT     ZBH
AFL     APO     BLK     CHD     CRWD    DLTR    EPAM    FIS     GNRC    HSY     IVZ     KVUE    MAS     MSCI    NVR     PH      RF      SOXL    TER     UBER    WBA     ZBRA
AIG     APTV    BMY     CHRW    CSCO    DOC     EQIX    FITB    GOOG    HUBB    IWM     L       MCD     MSFT    NWS     PHM     RJF     SOXX    TFC     UDR     WBD     ZTS
AIQ     ARE     BR      CHTR    CSGP    DOV     EQR     FMC     GOOGL   HUM     J       LABU    MCHP    MSI     NWSA    PKG     RL      SPG     TFX     UHS     WDAY
AIZ     ATO     BRK.B   CI      CSX     DOW     EQT     FOX     GPC     HWM     JBHT    LDOS    MCK     MTB     NXPI    PLD     RMD     SPGI    TGT     ULTA    WDC
AJG     AVB     BRO     CINF    CTAS    DPZ     ERIE    FOXA    GPN     IBIT    JBL     LEN     MCO     MTCH    O       PLTR    ROK     SPXL    TJX     UNH     WEC
AKAM    AVGO    BSX     CL      CTRA    DRI     ES      FRT     GRMN    IBM     JCI     LH      MDLZ    MTD     ODFL    PM      ROL     SPY     TLT     UNP     WELL
ALB     AVY     BWA     CLX     CTSH    DTE     ESS     FSLR    GS      ICE     JKHY    LHX     MDT     MU      OKE     PNC     ROP     SRE     TMO     UPS     WFC
ALGN    AWK     BX      CMCSA   CTVA    DUK     ETN     FTNT    GWW     IDXX    JNJ     LII     MET     NCLH    OMC     PNR     ROST    SSO     TMUS    URI     WM
ALL     AXON    BXP     CME     CVS     DVA     ETR     FTV     HAL     IEF     JNK     LIN     META    NDAQ    ON      PNW     RSG     STE     TNA     USB     WMB
ALLE    AXP     C       CMG     CVX     DVN     EVRG    FXI     HAS     IEI     JNPR    LKQ     MGM     NDSN    ORCL    PODD    RTX     STLD    TPL     V       WMT
"""



TickerConfigName = Literal["10_TICKERS", "30_TICKERS", "ANOTHER_10_TICKERS", "ALL_TICKERS"]

TICKER_PRESETS: Dict[TickerConfigName, List[str]] = {
    "ALL_TICKERS": ALL_TICKERS.split(),
    "ANOTHER_10_TICKERS": [
        "BRK.B",
        "UNH",
        "V",
        "MA",
        "AVGO",
        "LLY",
        "JPM",
        "XOM",
        "COST",
        "HD",
    ],
    "10_TICKERS": [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "JNJ",
        "XOM",
    ],
    "30_TICKERS": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "NVDA",
        "AMD",
        "INTC",
        "IBM",
        "AMZN",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "COST",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "UNH",
        "XOM",
        "CVX",
        "COP",
        "CAT",
        "BA",
        "GE",
        "VZ",
        "T",
    ],
}


@dataclass(frozen=True)
class TickerConfig:
    """Ticker universe preset chosen by name."""

    name: TickerConfigName

    @property
    def tickers(self) -> List[str]:
        return TICKER_PRESETS[self.name]

    @property
    def tickers_with_cash(self) -> List[str]:
        return ["Cash", *self.tickers]

    def __len__(self) -> int:
        return len(self.tickers)

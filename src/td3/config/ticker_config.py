"""
Ticker universe definitions and configuration for the TD3 training pipeline.

Provides named presets of stock tickers (e.g. 10, 30, or all available symbols)
and the TickerConfig dataclass used to select and access a ticker universe by name.

Author: Peter Likavec
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import random

ALL_TICKERS = """
A       AMAT    AZO     CAG     CMI     CZR     DXCM    EW      HBAN    IEX     JNUG    LLY     MHK     NEE     ORLY    POOL    RVTY    STT     TPR     VGT     WRB
AAPL        BA      CAH     CMS     D       EA      EXC     GD      HCA     IFF     JPM     LMT     MKC     NEM         PPG     SBAC    STX     TQQQ        WST
ABBV    AMD     BAC         CNC     DAL     EBAY    EXPD    GDDY    HD      IJR     K       LNT     MKTX    NFLX    OXY     PPL     SBUX    STZ     TRGP    VLO     WTW
    AME     BALL    CAT     CNP          ECL     EXPE    GDX     HES     INCY    KDP     LOW     MLM     NI      PANW    PRU     SCHW    SW      TRMB    WY
ABT     AMGN    BAX     CB      COF     DD      ED      EXR         HIG     INTC    KEY     LQD     MMC     NKE         PSA     SHW     SWK     TROW    VMC     WYNN
ACGL    AMP     BBY     CBOE    COO     DE      EFX     F       GE      HII     INTU    KEYS    LRCX    MMM     NLR     PAYC    PSX     SHY     SWKS    TRV     VOO     XEL
ACN     AMT     BDX     CBRE    COP     DECK    EG      FANG        HLT     INVH    KHC     LULU    MNST    NOC     PAYX    PTC     SILJ    SYF     TSCO    VRSK    XLK
ADBE    AMZN    BEN     CCI     COR     DELL    EIX     FAST    GEN     HOLX    IP      KIM     LUV     MO      NOW     PCAR    PWR     SJM     SYK     TSLA    VRSN    XLP
ADI     ANET    BF.B    CCL     COST    DFS     EL      FCX          HON     IPG     KKR     LVS     MOH     NRG     PCG     PYPL    SLB     SYY     TSN     VRTX    XLU
ADM     ANSS    BG      CDNS    CPAY    DG      ELV     FDS     GILD    HOOD    IQV     KLAC    LW      MOS     NSC     PEG     QCOM    SLV     T       TT          XLV
ADP     AON         CDW     CPB     DGX     EMB     FDX     GIS     HPE          KMB     LYB     MPC     NTAP    PEP     QLD        TAP     TTWO    VTR     XLY
ADSK    AOS     BK      CE      CPRT    DHI     EMN     FE      GL      HPQ     IRM     KMI     LYV     MPWR    NTRS    PFE     QQQ     SNA     TDG     TXN     VTRS    XOM
AEE     APA     BKNG    CEG     CPT     DHR     EMR     FFIV    GLD     HRL     ISRG    KMX     MA      MRK     NUE     PFG     RCL     SNPS    TDY     TXT     VZ      XYL
AEP     APD     BKR     CF      CRL     DIS     ENPH    FI      GLW     HSIC    IT      KO      MAA     MRNA    NUGT    PG      REG     SO      TECH    TYL     WAB     YUM
AES     APH     BLDR    CFG     CRM     DLR     EOG     FICO    GM      HST     ITW     KR      MAR     MS      NVDA    PGR     REGN        TEL     UAL     WAT     ZBH
AFL     APO     BLK     CHD     CRWD    DLTR    EPAM    FIS     GNRC    HSY     IVZ         MAS     MSCI    NVR     PH      RF      SOXL    TER     UBER    WBA     ZBRA
AIG     APTV    BMY     CHRW    CSCO    DOC     EQIX    FITB    GOOG    HUBB    IWM     L       MCD     MSFT    NWS     PHM     RJF     SOXX    TFC     UDR     WBD     ZTS
AIQ     ARE     BR      CHTR    CSGP    DOV     EQR     FMC     GOOGL   HUM     J       LABU    MCHP    MSI     NWSA    PKG     RL      SPG     TFX     UHS     WDAY
AIZ     ATO     BRK.B   CI      CSX     DOW     EQT     FOX     GPC     HWM     JBHT    LDOS    MCK     MTB     NXPI    PLD     RMD     SPGI    TGT     ULTA    WDC
AJG     AVB     BRO     CINF    CTAS    DPZ     ERIE    FOXA    GPN         JBL     LEN     MCO     MTCH    O       PLTR    ROK     SPXL    TJX     UNH     WEC
AKAM    AVGO    BSX     CL      CTRA    DRI     ES      FRT     GRMN    IBM     JCI     LH      MDLZ    MTD     ODFL    PM      ROL     SPY     TLT     UNP     WELL
ALB     AVY     BWA     CLX     CTSH    DTE     ESS     FSLR    GS      ICE     JKHY    LHX     MDT     MU      OKE     PNC     ROP     SRE     TMO     UPS     WFC
ALGN    AWK     BX      CMCSA   CTVA    DUK     ETN     FTNT    GWW     IDXX    JNJ     LII     MET     NCLH    OMC     PNR     ROST    SSO     TMUS    URI     WM
ALL     AXON    BXP     CME     CVS     DVA     ETR     FTV     HAL     IEF     JNK     LIN     META    NDAQ    ON      PNW     RSG     STE     TNA     USB     WMB
ALLE    AXP     C       CMG     CVX     DVN     EVRG    FXI     HAS     IEI     JNPR    LKQ     MGM     NDSN    ORCL    PODD    RTX     STLD    TPL     V       WMT
"""


LESS_TICKERS = """
A       AMAT    AZO     CAG     CMI     CZR     DXCM    EW      HBAN    IEX     JNUG    LLY     MHK     NEE     ORLY    POOL    RVTY    STT     TPR     VGT     WRB
AAPL        BA      CAH     CMS     D       EA      EXC     GD      HCA     IFF     JPM     LMT     MKC     NEM         PPG     SBAC    STX     TQQQ        WST
ABBV    AMD     BAC         CNC     DAL     EBAY    EXPD    GDDY    HD      IJR     K       LNT     MKTX    NFLX    OXY     PPL     SBUX    STZ     TRGP    VLO     WTW
    AME     BALL    CAT     CNP          ECL     EXPE    GDX     HES     INCY    KDP     LOW     MLM     NI      PANW    PRU     SCHW    SW      TRMB    WY
ABT     AMGN    BAX     CB      COF     DD      ED      EXR         HIG     INTC    KEY     LQD     MMC     NKE         PSA     SHW     SWK     TROW    VMC     WYNN
ACGL    AMP     BBY     CBOE    COO     DE      EFX     F       GE      HII     INTU    KEYS    LRCX    MMM     NLR     PAYC    PSX     SHY     SWKS    TRV     VOO     XEL
ACN     AMT     BDX     CBRE    COP     DECK    EG      FANG        HLT     INVH    KHC     LULU    MNST    NOC     PAYX    PTC     SILJ    SYF     TSCO    VRSK    XLK
ADBE    AMZN    BEN     CCI     COR     DELL    EIX     FAST    GEN     HOLX    IP      KIM     LUV     MO      NOW     PCAR    PWR     SJM     SYK     TSLA    VRSN    XLP
ADI     ANET    BF.B    CCL     COST    DFS     EL      FCX          HON     IPG     KKR     LVS     MOH     NRG     PCG     PYPL    SLB     SYY     TSN     VRTX    XLU
ADM     ANSS    BG      CDNS    CPAY    DG      ELV     FDS     GILD    HOOD    IQV     KLAC    LW      MOS     NSC     PEG     QCOM    SLV     T       TT          XLV
ADP     AON         CDW     CPB     DGX     EMB     FDX     GIS     HPE          KMB     LYB     MPC     NTAP    PEP     QLD        TAP     TTWO    VTR     XLY
ADSK    AOS     BK      CE      CPRT    DHI     EMN     FE      GL      HPQ     IRM     KMI     LYV     MPWR    NTRS    PFE     QQQ     SNA     TDG     TXN     VTRS    XOM
AEE     APA     BKNG    CEG     CPT     DHR     EMR     FFIV    GLD     HRL     ISRG    KMX     MA      MRK     NUE     PFG     RCL     SNPS    TDY     TXT     VZ      XYL
AEP     APD     BKR     CF      CRL     DIS     ENPH    FI      GLW     HSIC    IT      KO      MAA     MRNA    NUGT    PG      REG     SO      TECH    TYL     WAB     YUM
AES     APH     BLDR    CFG     CRM     DLR     EOG     FICO    GM      HST     ITW     KR      MAR     MS      NVDA    PGR     REGN        TEL     UAL     WAT     ZBH
"""


TickerConfigName = Literal[
    "10_TICKERS",
    "ANOTHER_10_TICKERS",
    "ALL_TICKERS",
    "LESS_TICKERS",
    "RANDOM_TICKERS",
]

TICKER_PRESETS: Dict[TickerConfigName, List[str]] = {
    "LESS_TICKERS": LESS_TICKERS.split(),
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
}


@dataclass(frozen=True)
class TickerConfig:
    """Represents a named ticker universe preset.

    Attributes:
        name: Identifier of the ticker preset to use.
        _random_tickers: Optional cached random ticker sample used only when
            ``name`` is ``"RANDOM_TICKERS"``.
    """

    name: TickerConfigName
    _random_tickers: Optional[List[str]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initializes a random ticker sample for the random preset.

        When ``name`` is ``"RANDOM_TICKERS"`` and no sample is provided,
        this method stores a 10-symbol random sample from ``LESS_TICKERS``.
        """
        if self.name == "RANDOM_TICKERS" and self._random_tickers is None:
            object.__setattr__(self, "_random_tickers", random.sample(LESS_TICKERS.split(), 10))

    @property
    def tickers(self) -> List[str]:
        """Returns the active ticker universe.

        Returns:
            List[str]: Randomly sampled tickers for ``"RANDOM_TICKERS"`` or
            the predefined preset tickers for all other names.
        """
        if self.name == "RANDOM_TICKERS":
            return self._random_tickers or []
        return TICKER_PRESETS[self.name]

    @property
    def tickers_with_cash(self) -> List[str]:
        """Returns active tickers prefixed with the synthetic cash asset.

        Returns:
            List[str]: A list where ``"Cash"`` is the first symbol followed by
            the current ticker universe.
        """
        return ["Cash", *self.tickers]

    def __len__(self) -> int:
        """Returns the number of active tickers.

        Returns:
            int: Length of the current ticker universe.
        """
        return len(self.tickers)
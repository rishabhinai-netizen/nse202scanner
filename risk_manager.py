"""
Risk Manager — Position Sizing, Target Calculation, Portfolio Heat
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PositionSize:
    shares: int
    position_value: float
    risk_amount: float
    risk_pct_of_capital: float
    pct_of_portfolio: float
    warnings: List[str]


@dataclass
class Targets:
    entry: float
    stop_loss: float
    risk_per_share: float
    t1: float
    t1_rr: float
    t2: float
    t2_rr: float
    t3: float
    t3_rr: float
    trailing_trigger: float


class RiskManager:
    """Position sizing and risk management engine."""
    
    # HARD RULES (NON-NEGOTIABLE)
    MAX_RISK_PER_TRADE_PCT = 2.0
    MAX_SINGLE_POSITION_PCT = 20.0
    MAX_PORTFOLIO_HEAT_PCT = 10.0
    MAX_SECTOR_EXPOSURE_PCT = 30.0
    MAX_POSITIONS = 10
    MAX_LOSS_PER_TRADE_PCT = 7.0
    TIME_STOP_DAYS = 10
    
    @staticmethod
    def calculate_position(capital: float, risk_pct: float, 
                           entry: float, stop_loss: float,
                           position_multiplier: float = 1.0) -> PositionSize:
        """
        Calculate position size based on risk.
        Formula: Shares = (Capital × Risk%) / (Entry - Stop)
        """
        risk_pct = min(risk_pct, RiskManager.MAX_RISK_PER_TRADE_PCT)
        risk_pct *= position_multiplier  # Market regime adjustment
        
        risk_amount = capital * (risk_pct / 100)
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share <= 0:
            return PositionSize(0, 0, 0, 0, 0, ["Invalid: stop loss must differ from entry"])
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        pct_of_portfolio = (position_value / capital) * 100 if capital > 0 else 0
        
        warnings = []
        
        # Check position size limit
        if pct_of_portfolio > RiskManager.MAX_SINGLE_POSITION_PCT:
            old_shares = shares
            shares = int((capital * RiskManager.MAX_SINGLE_POSITION_PCT / 100) / entry)
            position_value = shares * entry
            pct_of_portfolio = (position_value / capital) * 100
            warnings.append(f"Position capped from {old_shares} to {shares} shares (20% max rule)")
        
        # Check max loss
        max_loss_pct = (risk_per_share / entry) * 100
        if max_loss_pct > RiskManager.MAX_LOSS_PER_TRADE_PCT:
            warnings.append(f"Stop loss is {max_loss_pct:.1f}% away — consider tighter stop (max 7%)")
        
        actual_risk = shares * risk_per_share
        
        return PositionSize(
            shares=shares,
            position_value=round(position_value, 2),
            risk_amount=round(actual_risk, 2),
            risk_pct_of_capital=round((actual_risk / capital) * 100, 2) if capital > 0 else 0,
            pct_of_portfolio=round(pct_of_portfolio, 2),
            warnings=warnings,
        )
    
    @staticmethod
    def calculate_targets(entry: float, stop_loss: float, is_short: bool = False) -> Targets:
        """Calculate targets at 1.5R, 2.5R, and 4R."""
        risk = abs(entry - stop_loss)
        
        if is_short:
            t1 = round(entry - 1.5 * risk, 2)
            t2 = round(entry - 2.5 * risk, 2)
            t3 = round(entry - 4 * risk, 2)
            trailing = round(entry - 2 * risk, 2)
        else:
            t1 = round(entry + 1.5 * risk, 2)
            t2 = round(entry + 2.5 * risk, 2)
            t3 = round(entry + 4 * risk, 2)
            trailing = round(entry + 2 * risk, 2)
        
        return Targets(
            entry=entry,
            stop_loss=stop_loss,
            risk_per_share=round(risk, 2),
            t1=t1, t1_rr=1.5,
            t2=t2, t2_rr=2.5,
            t3=t3, t3_rr=4.0,
            trailing_trigger=trailing,
        )
    
    @staticmethod
    def portfolio_heat(positions: List[Dict], capital: float) -> Dict:
        """
        Calculate total portfolio heat.
        Each position: {"symbol": str, "entry": float, "stop": float, "shares": int}
        """
        total_risk = 0
        position_details = []
        sector_exposure = {}
        
        for pos in positions:
            risk_per_share = abs(pos["entry"] - pos["stop"])
            position_risk = risk_per_share * pos["shares"]
            position_value = pos["entry"] * pos["shares"]
            total_risk += position_risk
            
            sector = pos.get("sector", "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
            
            position_details.append({
                "symbol": pos["symbol"],
                "risk": round(position_risk, 2),
                "risk_pct": round((position_risk / capital) * 100, 2),
                "value": round(position_value, 2),
                "value_pct": round((position_value / capital) * 100, 2),
            })
        
        heat_pct = round((total_risk / capital) * 100, 2) if capital > 0 else 0
        
        warnings = []
        if heat_pct > RiskManager.MAX_PORTFOLIO_HEAT_PCT:
            warnings.append(f"⚠️ Portfolio heat {heat_pct}% exceeds {RiskManager.MAX_PORTFOLIO_HEAT_PCT}% limit!")
        
        if len(positions) > RiskManager.MAX_POSITIONS:
            warnings.append(f"⚠️ {len(positions)} positions exceeds {RiskManager.MAX_POSITIONS} max!")
        
        for sector, value in sector_exposure.items():
            sector_pct = (value / capital) * 100 if capital > 0 else 0
            if sector_pct > RiskManager.MAX_SECTOR_EXPOSURE_PCT:
                warnings.append(f"⚠️ {sector} exposure {sector_pct:.1f}% exceeds 30% limit!")
        
        return {
            "total_risk": round(total_risk, 2),
            "heat_pct": heat_pct,
            "max_heat": RiskManager.MAX_PORTFOLIO_HEAT_PCT,
            "positions": position_details,
            "position_count": len(positions),
            "sector_exposure": {k: round((v/capital)*100, 1) for k, v in sector_exposure.items()},
            "warnings": warnings,
            "status": "🟢 SAFE" if heat_pct <= 6 else ("🟡 MODERATE" if heat_pct <= 10 else "🔴 DANGER"),
        }

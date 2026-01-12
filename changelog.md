# Changelog & Rollback Guide

## How to Rollback
This project uses a step-by-step development approach.
1. Check `task.md` to see the sequence of features added.
2. Each major feature is a "Step".
3. To rollback, revert the code state to the completion of the desired previous Step.

## Version History
- **v0.1**: Initial Project Structure (FastAPI/React - DEPRECATED).
- **v0.2**: Backend Services (FMP, EODHD, YFinance).
- **v0.3**: Migration to Streamlit + Yahoo Finance.
- **v1.0**: Stable Release (Logic working).
- **v1.1**: UI Polish.
- **v1.2**: Premium UI Overhaul.
- **v1.3**: Rekruiting Style (Deep Navy).
- **v1.4**: Full Design Suite.
- **v1.5**: Layout Refactor.
    - **Full Width**: Data table now spans the full container for better readability.
    - **Vertical Flow**: Analysis section moved below results.
    - **Formatting**: Verified 2-decimal precision for Support/Resistance levels.

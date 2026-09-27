import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import build_panel,PANEL_VERSION
if __name__=="__main__":
    data=ResearchData(ROOT/"data/processed"/VERSION)
    text,cache=prepare_names(data,ROOT)
    build_panel(data,text,cache,ROOT/"data/processed"/PANEL_VERSION)
    print("Shared task panel and exhaustive R0 tree contract verification completed.")

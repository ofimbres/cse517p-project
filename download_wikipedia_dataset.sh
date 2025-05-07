#!/bin/bash

# filepath: /Users/oscarfimbres/uw-git/cse517p-project/download_wikipedia_dataset.sh

# Define the directory to save the datasets
SAVE_DIR="data/wikipedia"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Check if Python and the datasets library are installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python3 to proceed."
    exit 1
fi

if ! python -c "import datasets" &> /dev/null; then
    echo "The 'datasets' library is not installed. Installing it now..."
    pip install datasets || { echo "Failed to install the 'datasets' library. Exiting."; exit 1; }
fi

# List of all available configurations
CONFIGS=(
    "20231101.en"
    # '20231101.ab', '20231101.ace', '20231101.ady', '20231101.af', '20231101.als', '20231101.alt', '20231101.am', '20231101.ami', '20231101.an', '20231101.ang', '20231101.anp', '20231101.ar', '20231101.arc', '20231101.ary', '20231101.arz', '20231101.as', '20231101.ast', '20231101.atj', '20231101.av', '20231101.avk', '20231101.awa', '20231101.ay', '20231101.az', '20231101.azb', '20231101.ba', '20231101.ban', '20231101.bar', '20231101.bat-smg', '20231101.bcl', '20231101.be', '20231101.be-x-old', '20231101.bg', '20231101.bh', '20231101.bi', '20231101.bjn', '20231101.blk', '20231101.bm', '20231101.bn', '20231101.bo', '20231101.bpy', '20231101.br', '20231101.bs', '20231101.bug', '20231101.bxr', '20231101.ca', '20231101.cbk-zam', '20231101.cdo', '20231101.ce', '20231101.ceb', '20231101.ch', '20231101.chr', '20231101.chy', '20231101.ckb', '20231101.co', '20231101.cr', '20231101.crh', '20231101.cs', '20231101.csb', '20231101.cu', '20231101.cv', '20231101.cy', '20231101.da', '20231101.dag', '20231101.de', '20231101.din', '20231101.diq', '20231101.dsb', '20231101.dty', '20231101.dv', '20231101.dz', '20231101.ee', '20231101.el', '20231101.eml', '20231101.en', '20231101.eo', '20231101.es', '20231101.et', '20231101.eu', '20231101.ext', '20231101.fa', '20231101.fat', '20231101.ff', '20231101.fi', '20231101.fiu-vro', '20231101.fj', '20231101.fo', '20231101.fon', '20231101.fr', '20231101.frp', '20231101.frr', '20231101.fur', '20231101.fy', '20231101.ga', '20231101.gag', '20231101.gan', '20231101.gcr', '20231101.gd', '20231101.gl', '20231101.glk', '20231101.gn', '20231101.gom', '20231101.gor', '20231101.got', '20231101.gpe', '20231101.gu', '20231101.guc', '20231101.gur',
    # '20231101.guw', '20231101.gv', '20231101.ha', '20231101.hak', '20231101.haw', '20231101.he', '20231101.hi', '20231101.hif', '20231101.hr', '20231101.hsb', '20231101.ht', '20231101.hu', '20231101.hy', '20231101.hyw', '20231101.ia', '20231101.id', '20231101.ie', '20231101.ig', '20231101.ik', '20231101.ilo', '20231101.inh', '20231101.io', '20231101.is', '20231101.it', '20231101.iu', '20231101.ja', '20231101.jam', '20231101.jbo', '20231101.jv', '20231101.ka', '20231101.kaa', '20231101.kab', '20231101.kbd', '20231101.kbp', '20231101.kcg', '20231101.kg', '20231101.ki', '20231101.kk', '20231101.kl', '20231101.km', '20231101.kn', '20231101.ko', '20231101.koi', '20231101.krc', '20231101.ks', '20231101.ksh', '20231101.ku', '20231101.kv', '20231101.kw', '20231101.ky', '20231101.la', '20231101.lad', '20231101.lb', '20231101.lbe', '20231101.lez', '20231101.lfn', '20231101.lg', '20231101.li', '20231101.lij', '20231101.lld', '20231101.lmo', '20231101.ln', '20231101.lo', '20231101.lt', '20231101.ltg', '20231101.lv', '20231101.mad', '20231101.mai', '20231101.map-bms', '20231101.mdf', '20231101.mg', '20231101.mhr', '20231101.mi', '20231101.min', '20231101.mk', '20231101.ml', '20231101.mn', '20231101.mni', '20231101.mnw', '20231101.mr', '20231101.mrj', '20231101.ms', '20231101.mt', '20231101.mwl', '20231101.my', '20231101.myv', '20231101.mzn', '20231101.nah', '20231101.nap', '20231101.nds', '20231101.nds-nl', '20231101.ne', '20231101.new', '20231101.nia', '20231101.nl', '20231101.nn', '20231101.no', '20231101.nov', '20231101.nqo', '20231101.nrm', '20231101.nso', '20231101.nv', '20231101.ny', '20231101.oc', '20231101.olo', '20231101.om', '20231101.or', '20231101.os', '20231101.pa', '20231101.pag', '20231101.pam', '20231101.pap', '20231101.pcd', '20231101.pcm', '20231101.pdc', '20231101.pfl', '20231101.pi', '20231101.pih', '20231101.pl', '20231101.pms', '20231101.pnb', '20231101.pnt', '20231101.ps', '20231101.pt', '20231101.pwn', '20231101.qu', '20231101.rm', '20231101.rmy', '20231101.rn', '20231101.ro', '20231101.roa-rup', '20231101.roa-tara', '20231101.ru', '20231101.rue', '20231101.rw', '20231101.sa', '20231101.sah', '20231101.sat', '20231101.sc', '20231101.scn', '20231101.sco', '20231101.sd', '20231101.se', '20231101.sg',
    # '20231101.sh', '20231101.shi', '20231101.shn', '20231101.si', '20231101.simple', '20231101.sk', '20231101.skr', '20231101.sl', '20231101.sm', '20231101.smn', '20231101.sn', '20231101.so', '20231101.sq', '20231101.sr', '20231101.srn', '20231101.ss', '20231101.st', '20231101.stq', '20231101.su', '20231101.sv', '20231101.sw', '20231101.szl', '20231101.szy', '20231101.ta', '20231101.tay', '20231101.tcy', '20231101.te', '20231101.tet', '20231101.tg', '20231101.th', '20231101.ti', '20231101.tk', '20231101.tl', '20231101.tly', '20231101.tn', '20231101.to', '20231101.tpi', '20231101.tr', '20231101.trv', '20231101.ts', '20231101.tt', '20231101.tum', '20231101.tw', '20231101.ty', '20231101.tyv', '20231101.udm', '20231101.ug', '20231101.uk', '20231101.ur', '20231101.uz', '20231101.ve', '20231101.vec', '20231101.vep', '20231101.vi', '20231101.vls', '20231101.vo', '20231101.wa', '20231101.war', '20231101.wo', '20231101.wuu', '20231101.xal', '20231101.xh', '20231101.xmf', '20231101.yi', '20231101.yo', '20231101.za', '20231101.zea', '20231101.zh', '20231101.zh-classical', '20231101.zh-min-nan', '20231101.zh-yue', '20231101.zu'
)

# Loop through each configuration and download the dataset
for CONFIG in "${CONFIGS[@]}"; do
    echo "Downloading dataset for configuration: $CONFIG"
    python - <<EOF
from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "$CONFIG", split="train")
dataset.save_to_disk("$SAVE_DIR/$CONFIG")
print(f"Dataset for configuration $CONFIG downloaded and saved to $SAVE_DIR/$CONFIG")
EOF

    if [ $? -ne 0 ]; then
        echo "Failed to download dataset for configuration: $CONFIG"
    else
        echo "Successfully downloaded dataset for configuration: $CONFIG"
    fi
done

echo "All datasets download process complete."
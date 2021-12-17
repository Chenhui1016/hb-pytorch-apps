RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

ENV_NAME="test"

if ! (return 0 2>/dev/null) ; then
    # If return is used in the top-level scope of a non-sourced script,
    # an error message is emitted, and the exit code is set to 1
    echo
    echo -e $RED"This script should be sourced like"$NC
    echo "    . ./prepare_dataset.sh"
    echo
    exit 1  # we detected we are NOT source'd so we can use exit
fi

if type conda 2>/dev/null ; then
  if conda info --envs | grep ${ENV_NAME}; then
    echo -e $CYAN"activating environment ${ENV_NAME}"$NC
  else
    echo
    echo -e $RED"(!) Please install the conda environment ${ENV_NAME} using"$NC
    echo "    conda env create --file=conda_env.yml"
    echo
    return 1  # we are source'd so we cannot use exit
  fi
else
    echo
    echo -e $RED"(!) Please install anaconda"$NC
    echo
    return 1  # we are source'd so we cannot use exit
fi

conda activate ${ENV_NAME}

DOWNLOADS_DIR=data/1kg
mkdir -p ${DOWNLOADS_DIR}
cd ${DOWNLOADS_DIR}
echo -e $GREEN"Choose a chromosome (1-22 or X) to download and preprocess"$NC
read CHROMOSOME
echo -e $CYAN"downloading chromosome ${CHROMOSOME} vcf from 1kg project phase 3"$NC
VCF_GZ_FILE_NAME=ALL.chr${CHROMOSOME}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/${VCF_GZ_FILE_NAME}
cd ../..

echo -e $GREEN"Choose a lower bound for the SNPs you want to include in the processed dataset, for instance 1"$NC
read SUBSET_LOWER_BOUND
echo -e $GREEN"Choose an upper bound for the SNPs you want to include in the processed dataset, for instance 1000"$NC
read SUBSET_UPPER_BOUND
python extract_vcf_sublist.py ./${DOWNLOADS_DIR}/${VCF_GZ_FILE_NAME} $SUBSET_LOWER_BOUND $SUBSET_UPPER_BOUND
python maf_from_vcf.py ./${DOWNLOADS_DIR}/${VCF_GZ_FILE_NAME}

echo
echo -e $CYAN"Now you can run the model like this:"$NC
echo -e $GREEN"    python SCDA.py HAPS_PATH MAFS_PATH [...options]"$NC
echo -e "    or run ${GREEN}python SCDA.py --help${NC} first to see all available options"
echo

# conda deactivate

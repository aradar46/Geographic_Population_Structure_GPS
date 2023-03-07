# Population Genetics Project

The dataset of ancient Eurasians (see [DataS1](https://github.com/sarabehnamian/Origins-of-Ancient-Eurasian-Genomes/tree/main/steps/Step%200)). The dataset contains **genetic data** of **ancient Eurasians** from **different time periods**. The data is in the form of **SNP patterns**.

Develop an interface that allows selecting **populations** and **time** and calculates the **minor allele frequencies** of the chosen populations. The program will **merge patterns** that are too close. The results will be printed on the screen as **admixture plots**.


# PLINK 1.9

[PLINK](https://www.cog-genomics.org/plink/1.9/) is a free, open-source whole genome association analysis toolset, designed to perform a range of basic, large-scale analyses in a computationally efficient manner. The focus of PLINK is purely on analysis of genotype/phenotype data, so there is no support for a file format for storing sequence data or working with the sequence data directly.

[_More..._](https://zzz.bwh.harvard.edu/plink/)

[_Tutorial_](https://zzz.bwh.harvard.edu/plink/tutorial.shtml)

**_Installation_**

```bash

condainstall-cbiocondaplink

```

ver: plink-1.90b6.21

converting the data to ped format

```bash

plink--bfileDataS1--maf0.01--recode--outs1

```

> During the filtering process, 147229 variants were loaded from the .bim file and 961 individuals were loaded from the .fam file. Among them, 526 were males, 388 were females, and 47 had ambiguous sex. 935 phenotype values were loaded from the .fam file, with 26 missing phenotypes.

> After filtering, 118837 variants and 961 individuals passed the quality control (QC) checks. Among the remaining phenotypes, 935 were controls and 0 were cases. Finally, the command outputted the converted PED/MAP files named "s1.ped" and "s1.map".

the ped file is a tab-delimited text file with 6 columns:

1. Family ID
2. Individual ID
3. Paternal ID
4. ...

which allells come ....

and the map file is a tab-delimited text file with 4 columns:

1. Chromosome (or contig, scaffold, etc.)
2. rs# or snp identifier
3. Genetic distance (morgans)
4. Base-pair position (bp units)

## codes presented in the jupyter notebook

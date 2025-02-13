conda install maturin
cd environment
maturin dev --release
cargo build --bin peano --release
cd ../learning
conda install -r requirements.txt

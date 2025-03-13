conda install maturin
cd environment
maturin dev --release
cargo build --bin peano --release
cd ../learning
pip install -r requirements.txt

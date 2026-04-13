use model_rs::Result;

#[tokio::main]
async fn main() -> Result<()> {
    model_rs::run().await
}

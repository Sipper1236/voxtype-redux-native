use std::{
    io::{self, BufRead, Write},
    time::Instant,
};
use voxtype::{config::Config, transcribe};
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        anyhow::bail!("usage: bench_worker CONFIG.toml");
    }
    let cfg: Config = toml::from_str(&std::fs::read_to_string(&args[1])?)?;
    let start = Instant::now();
    let model = transcribe::create_transcriber(&cfg)?;
    println!(
        "{}",
        serde_json::json!({"ready":true,"load_seconds":start.elapsed().as_secs_f64()})
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let req: serde_json::Value = serde_json::from_str(&line?)?;
        let file = req["file"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("request requires a file path"))?;
        let mut w = hound::WavReader::open(file)?;
        let spec = w.spec();
        if spec.sample_rate != 16000
            || spec.channels != 1
            || spec.bits_per_sample != 16
            || spec.sample_format != hound::SampleFormat::Int
        {
            anyhow::bail!("expected mono PCM16 WAV at 16 kHz");
        }
        let pcm: Vec<f32> = w
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let start = Instant::now();
        let result = model.transcribe(&pcm);
        let response = match result {
            Ok(text) => {
                serde_json::json!({"ok":true,"text":text,"seconds":start.elapsed().as_secs_f64()})
            }
            Err(e) => serde_json::json!({"ok":false,"error":e.to_string()}),
        };
        println!("{response}");
        io::stdout().flush()?;
    }
    Ok(())
}

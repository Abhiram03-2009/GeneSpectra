'use client';

import React, { useState, useCallback } from 'react';
import { ClipLoader } from 'react-spinners';
import { FiClipboard, FiUpload, FiZap } from 'react-icons/fi';
import Modal from 'react-modal';
import { motion } from 'framer-motion';
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from 'tsparticles-engine';
import { particlesConfig } from '../particlesConfig';
import PredictionCard from '../components/PredictionCard';
import GradCamHeatmap from '../components/GradCamHeatmap';

// Interfaces for our data structure
interface PredictionData { label: string; confidence: number; }
interface Visualizations { panel_a: string; panel_b: string; panel_c: string; confusion_matrix: string; letter_encoded_output: string; base_specific_activations: string; letter_encoded_output_text: string; }
interface ChartData { activationProfile: any[]; baseActivations: any; }
interface Summary { gene: string; sequenceLength: string; keyFindings: string[]; performanceMetrics: Record<string, string>; clinicalSignificance: string[]; }
interface ApiResponse { prediction: PredictionData; visualizations: Visualizations; chartData: ChartData; summary: Summary; }

Modal.setAppElement('body');

const CommandDeckPage: React.FC = () => {
  const [sequence, setSequence] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const particlesInit = useCallback(async (engine: Engine) => { await loadSlim(engine); }, []);
  
  const handleSubmit = async () => { 
    if (sequence.trim().length < 10) { setError('Please enter a valid DNA sequence.'); return; }
    setIsLoading(true); setError(''); setResults(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dna_sequence: sequence }) });
      if (!response.ok) throw new Error((await response.json()).error || 'Prediction server error.');
      const data = await response.json();
      setResults(data);
    } catch (err: any) { setError(err.message); } finally { setIsLoading(false); }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { 
    const file = e.target.files?.[0]; if (!file) return; const reader = new FileReader();
    reader.onload = (event) => { let text = (event.target?.result as string) || ''; if (text.startsWith('>')) { text = text.split('\n').slice(1).join('').replace(/\s/g, ''); } setSequence(text.toUpperCase()); };
    reader.readAsText(file);
  };

  return (
    <div className="h-screen w-full p-4 md:p-6 lg:p-8 bg-black relative overflow-hidden">
      <Particles id="tsparticles" init={particlesInit} options={particlesConfig} className="fixed inset-0 -z-10" />

      <div className="fixed top-8 left-8 z-50">
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex items-center rounded-2xl bg-black/40 shadow-xl animate-border-glow px-6 py-4"
          style={{ minWidth: '340px', maxWidth: '420px' }}
        >
          {/* Logo on far left, vertically centered */}
          <div className="bg-black/40 rounded-full p-2 border border-cyan-300 shadow-cyan-300 animate-border-glow flex-shrink-0 mr-4">
            <img
              src="/Logo.png"
              alt="GeneSpectra Logo"
              className="w-12 h-12 object-contain"
              style={{ width: '80px', height: '80px' }}
            />
          </div>

          {/* Text block centered in remaining space */}
          <div className="flex-1 flex flex-col items-center justify-center">
            <span className="text-[20px] font-bold tracking-wider bg-clip-text text-white bg-gradient-to-r from-sky-300 to-cyan-300 text-center">
              GeneSpectra
            </span>
            <h5 className="text-sm font-calibri text-white/70 mt-1 text-center">
              AI-Powered Pathogenicity Analysis
            </h5>
          </div>
        </motion.div>
      </div>

      {/* Main content */}
      <div className="">
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-[45%] z-40 w-full max-w-[1900px] max-h-[2000px]">
          <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* LEFT COLUMN: Inputs */}
            {!results && (
              <motion.div
                initial={{ x: -50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ duration: 0.7, delay: 0.4 }}
                className="lg:col-span-1 flex flex-col gap-8"
              >
                {/* Manual Input Card */}
                <section className="hud-card relative overflow-hidden rounded-3xl border border-cyan-600/40 backdrop-blur-lg bg-gradient-to-br from-[#0a1e33]/90 to-[#142b43]/90 shadow-2xl p-8 hover:border-cyan-300 transition-all duration-300 group">
                  <div className="absolute inset-0 pointer-events-none opacity-10 group-hover:opacity-20 transition-opacity duration-300" style={{ background: 'radial-gradient(circle at top right, #0ea5e9 0%, transparent 50%), radial-gradient(circle at bottom left, #a855f7 0%, transparent 50%)' }}></div>
                  <header className="relative z-10 flex items-center gap-4 mb-6">
                    <FiClipboard size={32} className="text-cyan-400 drop-shadow-lg" />
                    <h2 className="text-3xl font-bold tracking-wide text-cyan-300 select-none">Manual Input</h2>
                  </header>
                  <textarea
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value.toUpperCase())}
                    spellCheck={false}
                    placeholder="Paste FASTA or raw sequence..."
                    className="relative z-10 w-full h-52 p-6 bg-transparent border border-cyan-500 rounded-xl font-mono text-base text-white placeholder-cyan-600 resize-none focus:outline-none focus:ring-4 focus:ring-cyan-500 transition-shadow shadow-inner"
                    style={{ color: 'white' }}
                  />
                </section>

                {/* Upload File Card */}
                <section className="hud-card relative overflow-hidden rounded-3xl border border-cyan-600/40 backdrop-blur-lg bg-gradient-to-br from-[#0a1e33]/90 to-[#142b43]/90 shadow-2xl p-8 hover:border-cyan-300 transition-all duration-300 group">
                  <div className="absolute inset-0 pointer-events-none opacity-10 group-hover:opacity-20 transition-opacity duration-300" style={{ background: 'radial-gradient(circle at top left, #a855f7 0%, transparent 50%), radial-gradient(circle at bottom right, #0ea5e9 0%, transparent 50%)' }}></div>
                  <header className="relative z-10 flex items-center gap-4 mb-6">
                    <FiUpload size={32} className="text-cyan-400 drop-shadow-lg" />
                    <h2 className="text-3xl font-bold tracking-wide text-cyan-300 select-none">Upload File</h2>
                  </header>
                  <label
                    className="relative z-10 w-full cursor-pointer flex items-center justify-center p-8 rounded-xl border-4 border-dashed border-cyan-500/50 hover:border-cyan-400 transition-colors text-cyan-300 font-semibold text-lg hover:text-white select-none shadow-md"
                  >
                    Click to upload (.fasta, .fa, .txt)
                    <input type="file" accept=".fasta,.fa,.txt" onChange={handleFileChange} className="hidden" />
                  </label>
                </section>

                {/* Analyze Button */}
                <button
                  disabled={isLoading}
                  onClick={handleSubmit}
                  className="relative z-10 w-full mt-6 py-5 bg-gradient-to-r from-cyan-600 to-sky-600 hover:from-cyan-500 hover:to-sky-500 rounded-3xl font-extrabold text-white text-2xl shadow-lg transition-transform hover:scale-[1.03] disabled:opacity-50 disabled:cursor-not-allowed transform-gpu animate-shimmer"
                >
                  {isLoading ? (
                    <ClipLoader color="white" size={32} />
                  ) : (
                    <div className="flex items-center justify-center gap-4" style={{ color: 'white' }}>
                      <FiZap size={32} />
                      Analyze Sequence
                    </div>
                  )}
                </button>

                {/* Error message */}
                {error && <p className="mt-4 text-red-400 text-center font-mono text-lg">{error}</p>}
              </motion.div>
            )}

            {/* RIGHT COLUMN: Results */}
            <motion.div
              initial={{ x: 50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.6 }}
              className={`${results ? 'lg:col-span-3' : 'lg:col-span-2'}`}
            >
              {results ? (
                <div className="flex flex-col gap-10">
                  <PredictionCard
                    prediction={results.prediction}
                    chartData={results.chartData}
                    summary={results.summary}
                    onViewVisuals={() => setIsModalOpen(true)}
                    confusionMatrixImage={results.visualizations.confusion_matrix}
                    letterEncodedOutputImage={results.visualizations.letter_encoded_output}
                    panelAImage={results.visualizations.panel_a}
                    panelBImage={results.visualizations.panel_b}
                    panelCImage={results.visualizations.panel_c}
                    baseSpecificActivationsImage={results.visualizations.base_specific_activations}
                    isViewVisualsDisabled={!results}
                    letterEncodedOutputText={results.visualizations.letter_encoded_output_text}
                  />
                </div>
              ) : (
                <section className="hud-card relative overflow-hidden w-full min-h-[520px] rounded-3xl flex flex-col items-center justify-center gap-6 text-center bg-gradient-to-br from-[#0a1e33]/80 to-[#142b43]/90 shadow-2xl border border-cyan-600/30 p-12 animate-shimmer">
                  <div className="absolute inset-0 pointer-events-none opacity-10" style={{ background: 'radial-gradient(circle at center, #0ea5e9 0%, transparent 70%)' }}></div>
                  <p className="relative z-10 text-8xl animate-pulse select-none">🧬</p>
                  <h2 className="relative z-10 text-4xl font-extrabold tracking-wide text-cyan-400">Awaiting Sequence Analysis</h2>
                  <p className="relative z-10 text-lg max-w-xl text-cyan-300 px-4">Paste or upload a DNA sequence, then click Analyze to see predictions and insights.</p>
                </section>
              )}
            </motion.div>
          </main>
        </div>
      </div>
      <Modal isOpen={isModalOpen} onRequestClose={() => setIsModalOpen(false)} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[95%] max-w-6xl h-[90vh] hud-card p-6 border-2 border-cyan-400/50 shadow-2xl" overlayClassName="fixed inset-0">
        {results && <GradCamHeatmap visualizations={results.visualizations} />}
      </Modal>
    </div>
  );
};

export default CommandDeckPage;
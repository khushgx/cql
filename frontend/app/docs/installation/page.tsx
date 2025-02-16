"use client"
import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function InstallationPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Installation</h1>
            <p className="text-lg text-white/70">Set up RL-CP Fusion in your environment</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">Follow these steps to install RL-CP Fusion and its dependencies.</p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">System Requirements</h2>
          <ul className="space-y-2 list-disc list-inside mb-8">
            <li className="text-white/70">Python 3.7 or higher</li>
            <li className="text-white/70">CUDA-compatible GPU (recommended)</li>
            <li className="text-white/70">64-bit operating system</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Installing Dependencies</h2>
          <p className="mb-4">First, install PyTorch with CUDA support:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`}</code>
            </pre>
          </div>

          <p className="mb-4">Install other required packages:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`pip install gym d4rl numpy tensorboardX`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Installing RL-CP Fusion</h2>
          <p className="mb-4">Clone the repository:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`git clone https://github.com/yourusername/rl-cp-fusion.git
cd rl-cp-fusion`}</code>
            </pre>
          </div>

          <p className="mb-4">Install the package:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`pip install -e .`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Verifying Installation</h2>
          <p className="mb-4">Run this simple test to verify the installation:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`python -c "from conformal_sac.agent_wrapper import SACAgent; print('Installation successful!')"`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Common Issues</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">CUDA Issues</h3>
          <p className="mb-4">If you encounter CUDA-related errors, verify your CUDA installation:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`python -c "import torch; print(torch.cuda.is_available())"`}</code>
            </pre>
          </div>

          <h3 className="text-xl font-semibold mt-8 mb-4">D4RL Dataset Access</h3>
          <p className="mb-4">
            If you have trouble accessing D4RL datasets, make sure you have the correct permissions:
          </p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`python -c "import d4rl; import gym; env = gym.make('halfcheetah-medium-v2')"`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Next Steps</h2>
          <p className="mb-4">After installation, you can:</p>
          <ul className="space-y-2 list-disc list-inside">
            <li className="text-white/70">
              Follow the{" "}
              <a href="/docs/quickstart" className="text-blue-400 hover:text-blue-300">
                Quickstart Guide
              </a>
            </li>
            <li className="text-white/70">
              Read about{" "}
              <a href="/docs/sacagent" className="text-blue-400 hover:text-blue-300">
                SACAgent
              </a>
            </li>
            <li className="text-white/70">
              Explore{" "}
              <a href="/docs/api/configuration" className="text-blue-400 hover:text-blue-300">
                Configuration Options
              </a>
            </li>
          </ul>
        </div>
      </div>
    </DocLayout>
  )
}


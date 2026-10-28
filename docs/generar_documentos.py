"""
Genera los dos documentos en Word (y su PDF con Word, actualizando el índice):
  docs/Documento_tecnico_DT-GDM-01.docx / .pdf
  docs/Documento_pruebas_DP-GDM-01.docx / .pdf
"""

import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import documento_pruebas  # noqa: E402
import documento_tecnico  # noqa: E402


def a_pdf(docx):
    """Abre el .docx en Word, actualiza el índice y los campos, guarda y exporta a PDF."""
    ps = f"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$d = $w.Documents.Open('{docx}')
foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
$d.Fields.Update() | Out-Null
$d.Save()
$d.ExportAsFixedFormat('{docx.with_suffix(".pdf")}', 17)
$d.Close($false); $w.Quit()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


if __name__ == "__main__":
    for mod, nombre in ((documento_tecnico, "Documento_tecnico_DT-GDM-01.docx"), (documento_pruebas, "Documento_pruebas_DP-GDM-01.docx")):
        ruta = AQUI / nombre
        mod.construir(ruta)
        print("escrito", ruta, flush=True)
        if "--sin-pdf" not in sys.argv:
            a_pdf(ruta)
            print("pdf", ruta.with_suffix(".pdf"), flush=True)

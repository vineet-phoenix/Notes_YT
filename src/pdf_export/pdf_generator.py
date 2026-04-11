from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path
from typing import Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import PDFGenerationError, handle_exception

logger = get_logger(__name__)

class PDFGenerator:
    """Generates PDF files from notes."""
    
    def __init__(self):
        self.export_dir = settings.EXPORTS_DIR
    
    @handle_exception
    def generate_notes_pdf(self, video_title: str, notes: str, 
                          metadata: Dict = None) -> Path:
        """Generate PDF from notes."""
        
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = self.export_dir / filename
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph(video_title, title_style))
        story.append(Spacer(1, 0.3*inch))
        
        if metadata:
            story.append(Paragraph("Video Information", styles['Heading2']))
            meta_data = [
                ['Duration', str(metadata.get('duration', 'N/A'))],
                ['Uploader', str(metadata.get('uploader', 'N/A'))],
                ['Date', str(metadata.get('upload_date', 'N/A'))],
                ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ]
            
            table = Table(meta_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Generated Notes", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for note in notes.split('\n'):
            if note.strip():
                story.append(Paragraph(note, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        logger.info(f"Generated PDF: {pdf_path}")
        
        return pdf_path

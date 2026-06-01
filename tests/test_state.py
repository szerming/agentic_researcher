from agentic_researcher.state import ReportSkeleton, ReportSkeletonSection


class TestReportSkeleton:
    def test_generate_outline(self):
        sec1 = ReportSkeletonSection(
            title="Section 1",
            level=1,
            bullets=["Bullet 1", "Bullet 2"],
            subsections=[
                ReportSkeletonSection(
                    title="Subsection 1.1",
                    level=2,
                    bullets=["Bullet 1.1", "Bullet 1.2"],
                    subsections=[
                        ReportSkeletonSection(
                            title="Subsubsection 1.1.1",
                            level=3,
                            bullets=["Bullet 1.1.1", "Bullet 1.1.2"],
                        )
                    ],
                )
            ],
        )

        sec2 = ReportSkeletonSection(
            title="Section 2",
            level=1,
            bullets=["Bullet 3", "Bullet 4"],
        )

        skeleton = ReportSkeleton(
            title="Test Report",
            sections=[sec1, sec2],
        )
        outline = skeleton.to_outline()

        print(f"\n----\n{outline}")

        assert len(outline) > 0
